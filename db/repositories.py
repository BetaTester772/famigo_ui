from __future__ import annotations

"""Lightweight repository helpers for relational CRUD/queries.

This module replaces the previous monolithic `db_manager.py` for
non-vector operations (users, groups, events, audit logs).

Vector (embedding) operations live in:
  app/services/rag/pgvector_adapter.py
"""

import logging
import uuid
from typing import Iterable, List, Optional , Dict
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from db.models import (
    User,
    GroupMember,
    Group,
    Event,
    VisibilityLevel,
    AuditAccess,
    visible_user_ids_subquery,
)

logger = logging.getLogger(__name__)



def get_or_create_group_by_name(db: Session, group_name: str) -> Group:
    """그룹명이 없으면 새로 만들고, 있으면 그대로 반환."""
    grp = db.execute(select(Group).where(Group.name == group_name)).scalar_one_or_none()
    if grp:
        return grp
    grp = Group(name=group_name)
    db.add(grp)
    db.commit()
    db.refresh(grp)
    logger.info("[repo.create_group] name=%s id=%s", group_name, grp.group_id)
    return grp


def ensure_group_membership(db: Session, group_id: int, user_id: int, role: str = "member") -> None:
    """해당 그룹에 사용자가 없으면 멤버로 추가. 이미 있으면 무시."""
    exists = db.execute(
        select(GroupMember).where(
            GroupMember.group_id == group_id,
            GroupMember.user_id == user_id,
        )
    ).scalar_one_or_none()
    if exists:
        return
    try:
        db.add(GroupMember(group_id=group_id, user_id=user_id, role=role))
        db.commit()
        logger.info("[repo.join_group] user=%s -> group=%s (%s)", user_id, group_id, role)
    except IntegrityError:
        db.rollback()

# ---------------------------------------------------------------------------
# Users & Groups
# ---------------------------------------------------------------------------

def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.get(User, user_id)

def get_users_by_ids(db: Session, ids: Iterable[int]) -> Dict[int, str]:
    ids = list(set(int(x) for x in ids))
    if not ids:
        return {}
    rows = db.execute(select(User.user_id, User.name).where(User.user_id.in_(ids))).all()
    mapping = {int(uid): name for (uid, name) in rows}
    return mapping

def get_user_by_face_id(db: Session, face_id: uuid.UUID | str) -> Optional[User]:
    if isinstance(face_id, str):
        try:
            face_id = uuid.UUID(face_id)
        except Exception:  # noqa: BLE001
            logger.warning("[repo.user_by_face] invalid UUID: %s", face_id)
            return None
    user = db.scalar(select(User).where(User.face_id == face_id))
    logger.info("[repo.user_by_face] face=%s -> user_id=%s", face_id, getattr(user, "user_id", None))
    return user


def get_group_member_ids(db: Session, user_id: int, *, include_self: bool = False) -> List[int]:
    """Return IDs of users who share a group with `user_id`.
    include_self=False (default) ensures the caller is NOT in the list.
    """
    subq = visible_user_ids_subquery(user_id)
    rows = db.execute(subq).scalars().all()
    ids = set(int(x) for x in rows)
    if not include_self and user_id in ids:
        ids.remove(user_id)
    out = sorted(ids)
    logger.info("[repo.get_group_member_ids] me=%s include_self=%s -> %s", user_id, include_self, out)
    return out

def get_group_member_ids_by_group_name(db: Session, user_id: int, group_name: str, *, include_self: bool = False) -> List[int]:
    """Return IDs of users in the **named group** only.
        보안상, 호출자(user_id)가 해당 그룹의 멤버가 아닌 경우 **빈 리스트**를 반환합니다.
        """
    grp = db.execute(select(Group).where(Group.name == group_name)).scalar_one_or_none()
    if not grp:
        logger.info("[repo.get_group_member_ids_by_group_name] group not found: %s", group_name)
        return []


    # Ensure the caller is a member of this group
    me_in = db.execute(
    select(GroupMember.user_id).where(
    GroupMember.group_id == grp.group_id,
    GroupMember.user_id == user_id,
    )
    ).first()
    if not me_in:
        logger.warning("[repo.get_group_member_ids_by_group_name] user %s not in group '%s'", user_id, group_name)
        return []


    rows = db.execute(
    select(GroupMember.user_id).where(GroupMember.group_id == grp.group_id)
    ).scalars().all()
    ids = set(int(x) for x in rows)
    if not include_self and user_id in ids:
        ids.remove(user_id)
    out = sorted(ids)
    logger.info("[repo.get_group_member_ids_by_group_name] me=%s group='%s' -> %s", user_id, group_name, out)
    return out

def get_groups_for_user(db: Session, user_id: int) -> List[Group]:
    stmt = (
        select(Group)
        .join(GroupMember, GroupMember.group_id == Group.group_id)
        .where(GroupMember.user_id == user_id)
        .order_by(Group.name.asc())
    )
    rows = db.execute(stmt).scalars().all()
    logger.info("[repo.groups_for_user] user=%s -> %s groups", user_id, len(rows))
    return rows

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def create_event(
    db: Session,
    *,
    owner_id: int,
    starts_at: datetime,
    location: Optional[str],
    description: Optional[str],
    visibility: VisibilityLevel = VisibilityLevel.SELF,
) -> Event:
    evt = Event(
        owner_id=owner_id,
        starts_at=starts_at,
        location=location,
        description=description,
        visibility=visibility,
    )
    db.add(evt)
    db.flush()  # assign PK
    logger.info("[repo.create_event] owner_id=%s event_id=%s", owner_id, evt.event_id)
    return evt


def list_visible_events(db: Session, me_id: int, *, limit: int = 50) -> List[Event]:
    owners = visible_user_ids_subquery(me_id)
    stmt = (
        select(Event)
        .where(Event.owner_id.in_(owners))
        .where(Event.visibility.in_([VisibilityLevel.GROUP, VisibilityLevel.PUBLIC]))
        .order_by(Event.starts_at.desc())
        .limit(limit)
    )
    rows = db.execute(stmt).scalars().all()
    logger.info("[repo.list_events] me=%s -> %s rows", me_id, len(rows))
    return rows

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def log_audit_access(db: Session, *, viewer_id: int, subject_id: int, detail: str | None = None) -> None:
    row = AuditAccess(viewer_id=viewer_id, subject_id=subject_id, detail=detail)
    db.add(row)
    db.flush()
    logger.info("[repo.audit] viewer=%s subject=%s detail=%s", viewer_id, subject_id, detail)


__all__ = [
    "get_user_by_id",
    "get_user_by_face_id",
    "get_group_member_ids",
    "get_groups_for_user",
    "create_event",
    "list_visible_events",
    "log_audit_access",
]
