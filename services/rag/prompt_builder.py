from __future__ import annotations

from typing import Iterable, Optional, Tuple


def build_guardrailed_prompt(
    *,
    user_query: str,
    context_lines: Iterable[str],
    me_name: Optional[str] = None,
    locale: str = "ko",
) -> Tuple[str, str]:
    """Return (system, user) messages for ChatCompletion.

    - Context lines MUST already contain explicit owner tags, e.g.:
      "- (0.83) owner_id=2 owner=철수 vis=group :: 7시 오피스 미팅 ..."
    - Guardrail:
      * 'owner'가 현재 사용자(me)가 아닌 항목은 '당신의 일정'이라고 단정하지 말 것
      * 그런 정보는 반드시 '철수의 일정'처럼 소유자를 명시해 답할 것
      * 모호하면 확인 질문을 선호 (필요 시)
    """
    me_label = me_name or "사용자"

    if locale == "ko":
        system = (
            "역할: 팀 메모리를 활용하는 조수.\n"
            "규칙:\n"
            f"1) 컨텍스트에서 owner가 {me_label}(현재 사용자)이 아닌 항목은 "
            "   절대 '당신의 일정/정보'로 단정하지 말 것.\n"
            "   → 반드시 '철수의 일정', '영희 메모'처럼 **소유자를 명시**해서 말할 것.\n"
            "2) owner가 사용자 본인인 항목만 '당신(또는 내)'로 지칭 가능.\n"
            "3) 모호하거나 충돌하는 경우, 짧게 사실을 분리해서 말하고 필요 시 확인 질문을 할 것.\n"
            "4) 불필요한 개인정보는 요약 수준으로만 언급.\n"
        )
        user = (
            "다음 컨텍스트를 참고해 질문에 답하세요. "
            "항상 각 사실의 소유자(owner)를 고려해 표현을 조심하세요.\n\n"
            "컨텍스트:\n"
            + "\n".join(context_lines)
            + "\n\n"
            f"질문: {user_query}\n"
            "답변:"
        )
    else:
        system = (
            "Role: assistant that leverages team-shared memory.\n"
            f"Rules:\n1) If an item is not owned by {me_label}, NEVER claim it as the user's own.\n"
            "   → Always attribute: 'Chulsoo has...', 'Younghee's note...', etc.\n"
            "2) Only items owned by the current user may be phrased as 'you/your'.\n"
            "3) If ambiguous, separate facts and ask a brief clarification if needed.\n"
            "4) Minimize unnecessary personal details.\n"
        )
        user = (
            "Use the following context to answer. Be careful with ownership.\n\n"
            "Context:\n"
            + "\n".join(context_lines)
            + "\n\n"
            f"Question: {user_query}\n"
            "Answer:"
        )
    return system, user
