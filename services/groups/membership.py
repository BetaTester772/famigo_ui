from __future__ import annotations
from typing import Optional, Callable
from sqlalchemy import select

from db.session import SessionLocal
from db.models import Group, GroupMember
from db import repositories as repo


def ensure_group_interactive(
    me_id: int,
    group_name: Optional[str],
    *,
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> Optional[str]:
    """
    CLI 전용 인터랙티브 헬퍼.
    - group_name이 없으면 None 반환(= 모든 소속 그룹 대상)
    - 그룹이 없으면 생성 여부 물어보고 생성+가입
    - 그룹은 있는데 내가 멤버가 아니면 가입 여부 물어보고 가입
    - 사용자가 거부하면 None 반환(= 특정 그룹 제한 없이 팀 검색)
    - 성공 시 입력한 group_name 그대로 반환
    """
    if not group_name:
        return None

    with SessionLocal() as db:
        grp = db.execute(select(Group).where(Group.name == group_name)).scalar_one_or_none()
        if not grp:
            ans = (ask(f"Group '{group_name}' not found. Create and join? (Y/n): ").strip().lower() or "y")
            if ans != "y":
                out("Proceeding without restricting to a specific group.")
                return None
            grp = repo.get_or_create_group_by_name(db, group_name)
            repo.ensure_group_membership(db, grp.group_id, me_id, role="member")
            out(f"Created group '{group_name}' and added you as member.")
            return group_name

        # 이미 그룹이 존재하는 경우: 내 가입 여부 확인
        me_in = db.execute(
            select(GroupMember).where(
                GroupMember.group_id == grp.group_id,
                GroupMember.user_id == me_id,
            )
        ).scalar_one_or_none()
        if not me_in:
            ans = (ask(f"You are not a member of '{group_name}'. Join now? (Y/n): ").strip().lower() or "y")
            if ans != "y":
                out("Proceeding without restricting to a specific group.")
                return None
            repo.ensure_group_membership(db, grp.group_id, me_id, role="member")
            out(f"Joined group '{group_name}'.")

        return group_name






# from app.services.groups.membership import ensure_group_interactive

# # ...
# elif target == "team":
#     group_name = input("group_name (optional, 엔터=모든 소속 그룹): ").strip() or None

#     없으면 생성/가입 여부 묻고, 거부 시 None으로 풀림
#     group_name = ensure_group_interactive(me_id, group_name)