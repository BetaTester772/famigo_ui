from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence
import logging

from sqlalchemy import select, literal
from sqlalchemy.orm import Session

from db.models import Embedding, VisibilityLevel, visible_user_ids_subquery

# LlamaIndex core types (0.10+)
try:
    from llama_index.core.vector_stores.types import (
        VectorStore, VectorStoreQuery, VectorStoreQueryResult,
    )
    from llama_index.core.vector_stores import (
        MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
    )
    from llama_index.core.schema import BaseNode, TextNode
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This module requires llama-index >= 0.10. Install it first (`pip install llama-index`)."
    ) from e

logger = logging.getLogger(__name__)


@dataclass
class _FilterSpec:
    me: Optional[int] = None
    include_self: bool = True
    visibility_in: Optional[Sequence[str]] = None
    text_ref_prefix: Optional[str] = None
    owner_id_in: Optional[Sequence[int]] = None
    exclude_owner_id: Optional[int] = None


class PgVectorEmbeddingsStore(VectorStore):
    """VectorStore backed by our `embeddings` table (SQLAlchemy) — cosine only."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._sf = session_factory
        self._op = "<=>"  # cosine distance operator

    # ------------------------------- API ---------------------------------

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        ids: List[str] = []
        with self._sf() as db:
            for node in nodes:
                emb = node.get_embedding()
                if emb is None:
                    raise ValueError("Node is missing embedding.")
                meta = dict(node.metadata or {})
                owner_id = int(meta.get("owner_id"))
                visibility = meta.get("visibility", VisibilityLevel.SELF.value)
                text_ref = meta.get("text_ref") or node.node_id or node.get_doc_id() or ""
                if not text_ref:
                    raise ValueError("text_ref is required")

                row = Embedding(
                    owner_id=owner_id,
                    visibility=VisibilityLevel(visibility),
                    text_ref=text_ref,
                    vector=emb,                           # pgvector
                )
                db.add(row)
                db.flush()
                ids.append(str(row.embedding_id))
                logger.info("[adapter.add] owner=%s vis=%s ref=%s id=%s",
                            owner_id, visibility, text_ref, row.embedding_id)
            db.commit()
        return ids

    def delete(self, doc_id: Optional[str] = None, **kwargs: Any) -> None:
        ids: Optional[Iterable[int]] = None
        if kwargs.get("ids"):
            ids = [int(x) for x in kwargs["ids"]]
        with self._sf() as db:
            if doc_id:
                db.query(Embedding).filter(Embedding.text_ref == doc_id).delete(synchronize_session=False)
            if ids:
                db.query(Embedding).filter(Embedding.embedding_id.in_(list(ids))).delete(synchronize_session=False)
            db.commit()

    # ------------------------------ internals -----------------------------

    def _parse_filters(self, q: "VectorStoreQuery") -> _FilterSpec:
        """공식(filters) 우선, 레거시(metadata_filters) 보조."""
        spec = _FilterSpec()

        # 1) 공식 MetadataFilters
        lf: Optional[MetadataFilters] = getattr(q, "filters", None)
        if lf and getattr(lf, "filters", None):
            for f in lf.filters:
                key = getattr(f, "key", None)
                op = getattr(f, "operator", None)
                val = getattr(f, "value", None)

                if key == "owner_id":
                    if op == FilterOperator.IN:
                        spec.owner_id_in = list(val) if isinstance(val, (list, tuple, set)) else [val]
                    elif op == FilterOperator.NIN:
                        # 우리 스펙은 단일 exclude_owner_id만 지원 → 리스트면 첫 값만 사용
                        if isinstance(val, (list, tuple, set)):
                            if val:
                                spec.exclude_owner_id = int(next(iter(val)))
                        elif val is not None:
                            spec.exclude_owner_id = int(val)

                elif key == "visibility":
                    if op == FilterOperator.IN:
                        spec.visibility_in = list(val) if isinstance(val, (list, tuple, set)) else [val]

                elif key == "text_ref":
                    # 접두사 매칭: TEXT_MATCH → LIKE 'prefix%'
                    if op == FilterOperator.TEXT_MATCH and isinstance(val, str):
                        # 사용자가 'prefix%'로 넘긴 경우 그대로, 아니면 접두사로 간주
                        spec.text_ref_prefix = val[:-1] if val.endswith("%") else val

        # 2) 레거시 metadata_filters (점진 이행)
        md = getattr(q, "metadata_filters", None)
        if isinstance(md, dict):
            # me/include_self는 공식 필터에 동등 키가 없어 유지
            spec.me = md.get("owner_id_me", spec.me)
            spec.include_self = bool(md.get("include_self", spec.include_self))
            spec.visibility_in = md.get("visibility_in", spec.visibility_in)
            spec.text_ref_prefix = md.get("text_ref_prefix", spec.text_ref_prefix)
            spec.owner_id_in = md.get("owner_id_in", spec.owner_id_in)
            spec.exclude_owner_id = md.get("exclude_owner_id", spec.exclude_owner_id)

        return spec

    def query(self, query: "VectorStoreQuery", **kwargs: Any) -> "VectorStoreQueryResult":
        qvec = query.query_embedding
        if qvec is None:
            raise ValueError("VectorStoreQuery.query_embedding is required")

        top_k = query.similarity_top_k or 8
        spec = self._parse_filters(query)

        with self._sf() as db:
            stmt = select(
                Embedding.embedding_id.label("id"),
                Embedding.owner_id,
                Embedding.visibility,
                Embedding.text_ref,
                (Embedding.vector.op(self._op)(literal(qvec))).label("dist"),  # cosine distance
            )

            # owner scope
            if spec.owner_id_in:
                stmt = stmt.where(Embedding.owner_id.in_(list(spec.owner_id_in)))
            else:
                owner_in_select = None
                if spec.me is not None:
                    peers = visible_user_ids_subquery(spec.me)
                    owner_in_select = select(literal(spec.me)).union(peers) if spec.include_self else peers
                if owner_in_select is not None:
                    stmt = stmt.where(Embedding.owner_id.in_(owner_in_select))

            if spec.exclude_owner_id is not None:
                stmt = stmt.where(Embedding.owner_id != spec.exclude_owner_id)

            # visibility
            if spec.visibility_in:
                stmt = stmt.where(Embedding.visibility.in_([VisibilityLevel(v) for v in spec.visibility_in]))

            if spec.text_ref_prefix:
                stmt = stmt.where(Embedding.text_ref.like(f"{spec.text_ref_prefix}%"))

            stmt = stmt.order_by(Embedding.vector.op(self._op)(literal(qvec))).limit(top_k)
            rows = db.execute(stmt).fetchall()

        ids: List[str] = []
        sims: List[float] = []
        nodes: List[BaseNode] = []

        for r in rows:
            ids.append(str(r.id))
            # cosine distance d ∈ [0,2] → similarity = max(0, 1 - d) ∈ [0,1]
            sim = max(0.0, 1.0 - float(r.dist))
            sims.append(sim)

            meta = {
                "owner_id": int(r.owner_id),
                "visibility": r.visibility.value if hasattr(r.visibility, "value") else str(r.visibility),
                "text_ref": r.text_ref,
            }
            text = r.content or r.text_ref or ""
            # node id로 text_ref를 사용하면 상위 계층에서 dedup에 유리
            node = TextNode(text=text, id_=r.text_ref, metadata=meta)
            nodes.append(node)

        logger.info("[adapter.query] rows=%s top_k=%s", len(ids), top_k)
        # 공식: metadatas 대신 nodes를 반환
        return VectorStoreQueryResult(ids=ids, similarities=sims, nodes=nodes)

    @classmethod
    def for_queries_only(cls, session_factory: Callable[[], Session]) -> "PgVectorEmbeddingsStore":
        return cls(session_factory)
