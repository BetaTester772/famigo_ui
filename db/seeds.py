"""Seed script for voice-face-rag-app

Creates a minimal, reproducible dataset used in demos/tests:
- Users: 철수(Chulsoo), 영희(Younghee)
- Group: demo-team with both as members
- Event: 영희의 오늘/내일 19:00 오피스 미팅 (visibility='group')
- Embedding: placeholder zero-vector for the event (size = VECTOR_DIM)

Idempotent: running multiple times will not duplicate rows.

Run:
  $ python -m app.db.seeds
or
  $ PYTHONPATH=. python app/db/seeds.py
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional

from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from db.session import SessionLocal, init_db
from db.models import (
    User,
    Group,
    GroupMember,
    Event,
    Embedding,
    VisibilityLevel,
    VECTOR_DIM,
)

KST = ZoneInfo("Asia/Seoul")


@dataclass
class SeedResult:
    chulsoo_id: int
    younghee_id: int
    group_id: int
    event_id: int
    embedding_id: Optional[int]


def _get_or_create_user(db: Session, name: str) -> User:
    user = db.scalar(select(User).where(User.name == name))
    if user:
        print(f"[seed] User exists: {name} (id={user.user_id})")
        return user
    user = User(name=name)
    db.add(user)
    db.flush()
    print(f"[seed] Created User: {name} (id={user.user_id})")
    return user


def _get_or_create_group(db: Session, name: str) -> Group:
    group = db.scalar(select(Group).where(Group.name == name))
    if group:
        print(f"[seed] Group exists: {name} (id={group.group_id})")
        return group
    group = Group(name=name)
    db.add(group)
    db.flush()
    print(f"[seed] Created Group: {name} (id={group.group_id})")
    return group


def _ensure_membership(db: Session, group_id: int, user_id: int, role: str = "member") -> None:
    gm = db.scalar(
        select(GroupMember).where(
            and_(GroupMember.group_id == group_id, GroupMember.user_id == user_id)
        )
    )
    if gm:
        print(f"[seed] Membership exists: group={group_id} user={user_id} role={gm.role}")
        return
    db.add(GroupMember(group_id=group_id, user_id=user_id, role=role))
    db.flush()
    print(f"[seed] Added membership: group={group_id} user={user_id} role={role}")


def _next_kst_evening(hour: int = 19) -> datetime:
    now_kst = datetime.now(KST)
    start = now_kst.replace(hour=hour, minute=0, second=0, microsecond=0)
    if start <= now_kst:
        start += timedelta(days=1)
    return start.astimezone(timezone.utc)


def _get_or_create_event(db: Session, owner_id: int) -> Event:
    # Look for an existing 19:00 office meeting near today/tomorrow
    start_utc = _next_kst_evening(19)
    start_lo = start_utc - timedelta(minutes=1)
    start_hi = start_utc + timedelta(minutes=1)

    evt = db.scalar(
        select(Event).where(
            and_(
                Event.owner_id == owner_id,
                Event.starts_at >= start_lo,
                Event.starts_at <= start_hi,
                Event.visibility == VisibilityLevel.GROUP,
            )
        )
    )
    if evt:
        print(
            f"[seed] Event exists: id={evt.event_id} owner={owner_id} starts_at={evt.starts_at}"
        )
        return evt

    evt = Event(
        owner_id=owner_id,
        starts_at=start_utc,
        location="삼성동 오피스",
        description="오피스 미팅",
        visibility=VisibilityLevel.GROUP,
    )
    db.add(evt)
    db.flush()
    print(
        f"[seed] Created Event: id={evt.event_id} owner={owner_id} starts_at={evt.starts_at}"
    )
    return evt


def _ensure_event_embedding(db: Session, owner_id: int, event: Event) -> Optional[Embedding]:
    text_ref = f"event:{event.event_id}"
    emb = db.scalar(
        select(Embedding).where(
            and_(Embedding.owner_id == owner_id, Embedding.text_ref == text_ref)
        )
    )
    if emb:
        print(f"[seed] Embedding exists: id={emb.embedding_id} text_ref={text_ref}")
        return emb

    # Placeholder zero-vector so that vector index/queries work in demos
    vec = [0.0] * VECTOR_DIM
    emb = Embedding(
        owner_id=owner_id,
        vector=vec,  # type: ignore[arg-type]
        text_ref=text_ref,
        visibility=event.visibility,
    )
    db.add(emb)
    db.flush()
    print(f"[seed] Created Embedding: id={emb.embedding_id} for {text_ref}")
    return emb


def seed() -> SeedResult:
    print("[seed] Initialising DB (ensuring extensions if possible)…")
    init_db()
    with SessionLocal() as db:  # type: ignore[operator]
        print("[seed] Begin transaction…")
        try:
            # Users
            chulsoo = _get_or_create_user(db, "철수")
            younghee = _get_or_create_user(db, "영희")

            # Group & membership
            team = _get_or_create_group(db, "demo-team")
            _ensure_membership(db, team.group_id, chulsoo.user_id, role="member")
            _ensure_membership(db, team.group_id, younghee.user_id, role="member")

            # Event (owned by Younghee)
            evt = _get_or_create_event(db, younghee.user_id)

            # Embedding placeholder for the event
            emb = _ensure_event_embedding(db, younghee.user_id, evt)

            db.commit()
            print("[seed] Commit complete.")

            return SeedResult(
                chulsoo_id=chulsoo.user_id,
                younghee_id=younghee.user_id,
                group_id=team.group_id,
                event_id=evt.event_id,
                embedding_id=emb.embedding_id if emb else None,
            )
        except Exception as e:  # noqa: BLE001
            db.rollback()
            print(f"[seed] ERROR: {e}. Rolled back.")
            raise


if __name__ == "__main__":
    res = seed()
    print("[seed] Done:")
    print(
        f"  users: chulsoo={res.chulsoo_id}, younghee={res.younghee_id}\n"
        f"  group: demo-team={res.group_id}\n"
        f"  event: id={res.event_id}\n"
        f"  embedding: id={res.embedding_id}"
    )
