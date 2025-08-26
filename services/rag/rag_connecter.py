from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.session import SessionLocal
from db.models import Embedding
from db import repositories as repo

from services.embeddings.generator import get_embedder
from services.rag.retriever import PastQueryRAGRetriever
from services.rag.prompt_builder import build_guardrailed_prompt
from services.rag.llm_client import complete as llm_complete

logger = logging.getLogger(__name__)

_retriever = PastQueryRAGRetriever(get_embedder())
import subprocess
import os
from pathlib import Path
import ctypes

def phi35_chat(user_input, sys_prompt):
    os.chdir(Path("C:/Users/Qualcomm/workspace/famigo/phi/genie_bundle"))
    # sys_prompt = "You are a helpful assistant. Be helpful but brief."
    prompt = f"<|system|>{sys_prompt}<|end|><|user|>{user_input}<|end|><|assistant|>"

    try:
        result = subprocess.run(
            ["./genie-t2t-run.exe", "-c", "genie_config.json", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            output = result.stdout
            if "[BEGIN]:" in output and "[END]" in output:
                start = output.find("[BEGIN]:") + 8
                end = output.find("[END]")
                response = output[start:end].strip()
                return response
            return "응답 파싱 실패"
        else:
            return f"실행 오류: {result.stderr}"

    except subprocess.TimeoutExpired:
        return "응답 시간 초과"
    except Exception as e:
        return f"오류: {e}"
    finally:
        os.chdir("C:\\Users\Qualcomm\workspace\\famigo\\famigo_ui\\")

def _resolve_texts_by_ref(db: Session, refs: List[str]) -> Dict[str, str]:
    if not refs:
        return {}
    rows = db.execute(
        select(Embedding.text_ref, Embedding.content).where(Embedding.text_ref.in_(refs))
    ).all()
    return {r[0]: (r[1] or "") for r in rows}


def get_rag_response(
    user_id: int,
    query: str,
    *,
    target: str = "self",                 # 'self' | 'team' | 'user'
    subject_user_id: Optional[int] = None,
    group_name: Optional[str] = None,
    k_peers: int = 8,
    k_self: int = 5,
    peer_penalty: float = 0.90,
    min_peer_sim: float = 0.35,
    model_name: str = "gpt-4o-mini",
    save_query: bool = False,
) -> str:
    # 1) retrieve with guardrails on scope
    hits = _retriever.search_by_target(
        me_id=user_id,
        query_text=query,
        target=target,
        subject_user_id=subject_user_id,
        k_peers=k_peers,
        k_self=k_self,
        peer_penalty=peer_penalty,
        min_peer_sim=min_peer_sim,
        group_name=group_name,
    )

    # 2) hydrate contents
    text_refs = [h.get("text_ref", "") for h in hits if h.get("text_ref")]
    with SessionLocal() as db:
        ref_to_text = _resolve_texts_by_ref(db, text_refs)
        owner_ids = {int(h.get("owner_id")) for h in hits if h.get("owner_id") is not None}
        owner_map = repo.get_users_by_ids(db, owner_ids)

    # 3) build context lines with explicit owner tags
    lines: List[str] = []
    for h in hits:
        owner_id = int(h.get("owner_id", 0))
        owner_name = owner_map.get(owner_id, f"user:{owner_id}")
        vis = h.get("visibility", "group")
        sim = float(h.get("similarity", 0.0))
        ref = h.get("text_ref", "")
        content = (ref_to_text.get(ref) or ref or "")[:400]
        lines.append(f"- ({sim:.3f}) owner_id={owner_id} owner={owner_name} vis={vis} :: {content}")

    # 4) guarded prompt
    me_name = owner_map.get(user_id, "사용자")
    system, user = build_guardrailed_prompt(user_query=query, context_lines=lines, me_name=me_name)
    prompt = user  # we send system+user separately below

    # 5) call LLM
    # answer = llm_complete(model_name, prompt, system=system, max_tokens=600) #TODO: on-device-llm(sLM)으로 교체 예정
    answer  = phi35_chat(prompt, system)

    # 6) optionally save current query (default group)
    if save_query and query.strip():
        try:
            _retriever.save_past_query(user_id, query, visibility="group")
        except Exception as e:
            logger.warning("[rag.save_query] failed: %s", e)

    return answer


# TODO: answer 그대로 유저에게 표시(TTS)
# answer = get_rag_response(
#     user_id=me_id, #TODO: int type. dict로 중간 변환
#     query="오늘 7시에 오피스 미팅 있나?", # TODO: 쿼리
#     target="team",
#     group_name="Demo Team",  # 얼굴 인식으로 받은 그룹명 TODO: str type
# )

# TODO: 대화 요약 저장은 State.BYE일때.
# from conversation_summarizer import summarize_and_store
# res = summarize_and_store(me_id=me_id, messages=transcript, visibility=default_visibility)
# print(f"(conversation saved) text_ref={res['text_ref']} id={res['embedding_id']}")
