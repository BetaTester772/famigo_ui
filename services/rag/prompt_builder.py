from __future__ import annotations

from typing import Iterable, Optional, Tuple

def build_guardrailed_prompt(
    *,
    user_query: str,
    context_lines: Iterable[str],
    me_name: Optional[str] = None,
) -> Tuple[str, str]:

    """Return (system, user) messages for ChatCompletion.

    - Context lines MUST already contain explicit owner tags, e.g.:
      "- (0.83) owner_id=2 owner=철수 vis=group :: 7시 오피스 미팅 ..."
    - Guardrail:
      * 'owner'가 현재 사용자(me)가 아닌 항목은 '당신의 일정'이라고 단정하지 말 것
      * 그런 정보는 반드시 '철수의 일정'처럼 소유자를 명시해 답할 것
      * 모호하면 확인 질문을 선호 (필요 시)
    """
    
    me_label = me_name or "user"

    system = (
        "Role: assistant that leverages team-shared memory.\n"
        +f"Rules:\n1) If an item is not owned by {me_label}, NEVER claim it as the user's own.\n"
        # +"   → Always attribute: 'Chulsoo has...', 'Younghee's note...', etc.\n"
        +"2) Only items owned by the current user may be phrased as 'you/your'.\n"
        +"3) You must NEVER report other people’s info as the user's.\n"
        +"4) Always name the owner when citing others’ info.\n"
        +"5) If context conflicts, prefer the user's own records.\n"
        +"6) If ambiguous, separate facts and ask a brief clarification if needed.\n"
        +"7) Minimize unnecessary personal details.\n"

    )
    user = (
        "Use the following context to answer. Be careful with ownership.\n\n"
        +"Context:\n"
        + "\n".join(context_lines)
        + "\n\n"
        +f"Question: {user_query}\n"
        "Answer:"
    )
    return system, user
