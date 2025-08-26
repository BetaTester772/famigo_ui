from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence
from uuid import uuid4

from llama_index.core.vector_stores import (
    MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.schema import TextNode

from db.session import SessionLocal
from db import repositories as repo
from services.rag.pgvector_adapter import PgVectorEmbeddingsStore

logger = logging.getLogger(__name__)


class PastQueryRAGRetriever:
    """High-level retriever enforcing scope/guardrails for past-query search."""

    def __init__(
            self,
            embed_fn: Callable[[str], List[float]],
            *,
            text_ref_prefix: str = "query:",
            session_factory=SessionLocal,
    ) -> None:
        self._embed_fn = embed_fn
        self._store = PgVectorEmbeddingsStore(session_factory)  # cosine-only
        self._prefix = text_ref_prefix

    # ------------------------------- SAVE --------------------------------

    def save_past_query(
            self,
            me_id: int,
            text: str,
            *,
            visibility: str = "group",  # default team-shared memory
            text_ref: Optional[str] = None,
    ) -> str:
        vec = self._embed_fn(text)
        text_ref = text_ref or f"{self._prefix}{me_id}:{uuid4().hex}"
        node = TextNode(
            text=text,
            metadata={"owner_id": me_id, "visibility": visibility, "text_ref": text_ref},
            embedding=vec,
        )
        ids = self._store.add([node])
        eid = ids[0]
        logger.info("[retriever.save] me=%s vis=%s ref=%s id=%s", me_id, visibility, text_ref, eid)
        return eid

    # ------------------------------ UTIL ---------------------------------

    @staticmethod
    def _build_llama_filters(
            *,
            owner_id_in: Optional[Sequence[int]],
            visibility_in: Sequence[str],
            text_ref_prefix: Optional[str],
            exclude_owner_id: Optional[int],
    ) -> Optional[MetadataFilters]:
        """공식 MetadataFilters 구성."""
        filters: List[MetadataFilter] = []

        if owner_id_in:
            filters.append(MetadataFilter(
                key="owner_id", operator=FilterOperator.IN, value=list(owner_id_in)
            ))

        if visibility_in:
            filters.append(MetadataFilter(
                key="visibility", operator=FilterOperator.IN, value=list(visibility_in)
            ))

        if text_ref_prefix:
            # TEXT_MATCH → (pgvector 어댑터 등에서) LIKE 'prefix%'
            filters.append(MetadataFilter(
                key="text_ref", operator=FilterOperator.TEXT_MATCH, value=f"{text_ref_prefix}%"
            ))

        if exclude_owner_id is not None:
            filters.append(MetadataFilter(
                key="owner_id", operator=FilterOperator.NIN, value=[exclude_owner_id]
            ))

        return MetadataFilters(filters=filters, condition=FilterCondition.AND) if filters else None

    # ------------------------------ SEARCH -------------------------------

    def _search(
            self,
            qvec: List[float],
            *,
            top_k: int,
            owner_id_in: Optional[Sequence[int]],
            visibility_in: Sequence[str],
            text_ref_prefix: Optional[str] = None,
            exclude_owner_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        # ✅ 공식 필터 구성
        llama_filters = self._build_llama_filters(
            owner_id_in=owner_id_in,
            visibility_in=visibility_in,
            text_ref_prefix=text_ref_prefix,
            exclude_owner_id=exclude_owner_id,
        )

        q = VectorStoreQuery(
            query_embedding=qvec,
            similarity_top_k=top_k,
            filters=llama_filters,  # ← 공식
        )

        # (임시 호환) 구 어댑터가 q.filters 대신 q.metadata_filters를 읽는 경우를 대비
        # 어댑터 정리 완료되면 아래 6줄은 제거하세요.
        try:
            setattr(q, "metadata_filters", {
                "owner_id_in": owner_id_in,
                "visibility_in": list(visibility_in) if visibility_in else None,
                "text_ref_prefix": text_ref_prefix,
                "exclude_owner_id": exclude_owner_id,
            })
        except Exception:
            pass

        res = self._store.query(q)

        hits = [
            {
                "embedding_id": res.ids[i],
                "similarity": float(res.similarities[i]),
                **(res.metadatas[i] or {}),
            }
            for i in range(len(res.ids))
        ]
        return hits

    def search_self(self, me_id: int, query_text: str, *, k: int = 8) -> List[Dict[str, Any]]:
        qvec = self._embed_fn(query_text)
        hits = self._search(
            qvec,
            top_k=k,
            owner_id_in=[me_id],
            visibility_in=["self", "group"],  # my private + my group-shared
            text_ref_prefix=self._prefix,
        )
        for h in hits:
            h["source"] = "self"
        return hits

    def search_group_split(
            self,
            me_id: int,
            query_text: str,
            *,
            k_peers: int = 8,
            k_self: int = 5,
            peer_penalty: float = 0.90,  # down-weight peers a bit
            min_peer_sim: float = 0.35,  # ignore weak peer matches
            group_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        qvec = self._embed_fn(query_text)

        # peers (exclude me), group visibility only
        with SessionLocal() as db:
            if group_name:
                peer_ids = repo.get_group_member_ids_by_group_name(
                    db, me_id, group_name, include_self=False
                )
            else:
                peer_ids = repo.get_group_member_ids(db, me_id, include_self=False)

        peers_hits: List[Dict[str, Any]] = []
        if peer_ids:
            peers_hits = self._search(
                qvec,
                top_k=k_peers,
                owner_id_in=peer_ids,
                visibility_in=["group"],  # do NOT read their 'self'
                text_ref_prefix=self._prefix,
            )
            filtered = []
            for h in peers_hits:
                sim = float(h["similarity"]) * peer_penalty
                if sim >= min_peer_sim:
                    h["similarity"] = sim
                    h["source"] = "peers"
                    filtered.append(h)
            peers_hits = filtered

        # self bucket
        self_k = k_self if peer_ids else 8
        self_hits = self.search_self(me_id, query_text, k=self_k)

        # combine (keep order by similarity)
        all_hits = peers_hits + self_hits
        # dedup by text_ref (or embedding_id)
        seen = set()
        uniq = []
        for h in sorted(all_hits, key=lambda x: x["similarity"], reverse=True):
            key = h.get("text_ref") or h.get("embedding_id")
            if key in seen:
                continue
            seen.add(key)
            uniq.append(h)
        return uniq

    def search_by_target(
            self,
            me_id: int,
            query_text: str,
            *,
            target: str = "self",  # 'self' | 'team' | 'user'
            subject_user_id: Optional[int] = None,
            k_peers: int = 8,
            k_self: int = 5,
            peer_penalty: float = 0.90,
            min_peer_sim: float = 0.35,
            group_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        target = (target or "self").lower()
        if target == "self":
            return self.search_self(me_id, query_text, k=8)

        if target == "team":
            return self.search_group_split(
                me_id, query_text,
                k_peers=k_peers, k_self=k_self,
                peer_penalty=peer_penalty, min_peer_sim=min_peer_sim,
                group_name=group_name,
            )

        if target == "user" and subject_user_id:
            qvec = self._embed_fn(query_text)
            owner_ids = [subject_user_id]
            # if subject != me → only group-visible docs are allowed
            vis = ["self", "group"] if subject_user_id == me_id else ["group"]
            hits = self._search(
                qvec,
                top_k=8,
                owner_id_in=owner_ids,
                visibility_in=vis,
                text_ref_prefix=self._prefix,
            )
            for h in hits:
                h["source"] = "subject"
            return hits

        # default fallback
        return self.search_self(me_id, query_text, k=8)
