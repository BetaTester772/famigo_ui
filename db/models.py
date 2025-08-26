from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, List

from sqlalchemy import (
    DateTime,
    Enum as PgEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# PostgreSQL UUID / JSONB
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict

# Vector type from pgvector sqlalchemy helper
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class VisibilityLevel(str, Enum):
    SELF = "self"
    GROUP = "group"
    PUBLIC = "public"

    def __str__(self) -> str:
        return self.value


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)

    # ✅ JSONB + MutableDict (in-place 변경 추적) + 올바른 server_default
    profile_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        MutableDict.as_mutable(JSONB),
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )

    # Relationships
    groups: Mapped[List["GroupMember"]] = relationship(back_populates="user")
    events: Mapped[List["Event"]] = relationship(back_populates="owner")
    embeddings: Mapped[List["Embedding"]] = relationship(back_populates="owner")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User id={self.user_id} name={self.name!r}>"


class Group(Base):
    __tablename__ = "groups"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)

    members: Mapped[List["GroupMember"]] = relationship(back_populates="group")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Group id={self.group_id} name={self.name!r}>"


class GroupMember(Base):
    __tablename__ = "group_members"
    __table_args__ = (
        UniqueConstraint("group_id", "user_id", name="uq_group_user"),
        Index("ix_group_members_user_id", "user_id"),
    )

    group_id: Mapped[int] = mapped_column(
        ForeignKey("groups.group_id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(50), default="member", nullable=False)
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    group: Mapped[Group] = relationship(back_populates="members")
    user: Mapped[User] = relationship(back_populates="groups")


class Event(Base):
    __tablename__ = "events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    location: Mapped[Optional[str]] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(Text)
    visibility: Mapped[VisibilityLevel] = mapped_column(
        PgEnum(VisibilityLevel, name="visibility_level"),
        default=VisibilityLevel.SELF,
        nullable=False,
    )

    owner: Mapped[User] = relationship(back_populates="events")

    __table_args__ = (Index("ix_events_owner_visibility", "owner_id", "visibility"),)


VECTOR_DIM = 1536  # 벡터 차원


class Embedding(Base):
    __tablename__ = "embeddings"

    embedding_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    vector: Mapped[List[float]] = mapped_column(Vector(VECTOR_DIM), nullable=False)
    text_ref: Mapped[str] = mapped_column(Text, nullable=False)
    visibility: Mapped[VisibilityLevel] = mapped_column(
        PgEnum(VisibilityLevel, name="visibility_level", create_type=False),  # reuse enum
        default=VisibilityLevel.SELF,
        nullable=False,
    )

    owner: Mapped[User] = relationship(back_populates="embeddings")

    __table_args__ = (Index("ix_embeddings_owner_visibility", "owner_id", "visibility"),)


class AuditAccess(Base):
    __tablename__ = "audit_access"

    access_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    viewer_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.user_id"))
    subject_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.user_id"))
    access_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    detail: Mapped[Optional[str]] = mapped_column(Text)

    viewer: Mapped[User] = relationship("User", foreign_keys=[viewer_id])
    subject: Mapped[User] = relationship("User", foreign_keys=[subject_id])


def visible_user_ids_subquery(user_id: int):
    """Return a SQLAlchemy selectable yielding IDs visible to *user_id* (self + same group)."""
    from sqlalchemy import select

    gm1 = select(GroupMember.group_id).where(GroupMember.user_id == user_id).subquery()

    return select(GroupMember.user_id).where(GroupMember.group_id.in_(gm1)).distinct()
# ---------------------------------------------------------------------------
# EOF
