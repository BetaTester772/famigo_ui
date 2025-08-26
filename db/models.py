from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum


from sqlalchemy import (
    DateTime,
    Enum as PgEnum,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    Index,
    func,
)

#postgresql UUID
from sqlalchemy.dialects.postgresql import UUID
#DB 관리를 위한 ORM
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship 
# Vector type from pgvector?sqlalchemy helper
from pgvector.sqlalchemy import Vector






class Base(DeclarativeBase):

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default= lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False
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
    face_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False
    )

    name: Mapped[str] = mapped_column(String(100), nullable=False, unique = True)
    profile_json: Mapped[dict] = mapped_column(
        "profile_json", default=dict, server_default="{}", nullable=False
    )

    # Relationships
    groups: Mapped[list["GroupMember"]] = relationship(back_populates="user")
    events: Mapped[list["Event"]] = relationship(back_populates="owner")
    embeddings: Mapped[list["Embedding"]] = relationship(back_populates="owner")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<User id={self.user_id} name={self.name!r}>"

class Group(Base):
    __tablename__ = "groups"
    # group_id : PK
    group_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 그룹 이름
    name: Mapped[str] = mapped_column(String(100), nullable=False , unique = True)

    members: Mapped[list["GroupMember"]] = relationship(back_populates="group")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Group id={self.group_id} name={self.name!r}>"

# 중간테이블
class GroupMember(Base):
    __tablename__ = "group_members"
    __table_args__ = (
        UniqueConstraint("group_id", "user_id", name="uq_group_user"),
        Index("ix_group_members_user_id", "user_id"),
    )

    group_id: Mapped[int] = mapped_column(ForeignKey("groups.group_id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    #각각을 외래키로, 묶어서 복합 기본키로 사용함
    role: Mapped[str] = mapped_column(String(50), default="member", nullable=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    group: Mapped[Group] = relationship(back_populates="members")
    user: Mapped[User] = relationship(back_populates="groups")


#RAG의 근거 데이터(원문). 임베딩(embeddings)은 여기 내용을 벡터화한 인덱스 사본일 뿐.
#영희가 “오늘 19:00 삼성동 오피스 미팅”을 남기면 events에 1행 생성 → 별도로 embeddings에 임베딩 저장.
class Event(Base):
    __tablename__ = "events"

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    location: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    # 공유범위 설정
    visibility: Mapped[VisibilityLevel] = mapped_column(
        PgEnum(VisibilityLevel, name="visibility_level"), default=VisibilityLevel.SELF, nullable=False
    )

    owner: Mapped[User] = relationship(back_populates="events")

    __table_args__ = (
        Index("ix_events_owner_visibility", "owner_id", "visibility"),
    )




VECTOR_DIM = 1536  #벡터의 차원 설정


class Embedding(Base):
    __tablename__ = "embeddings"

    embedding_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    vector: Mapped[list[float]] = mapped_column(Vector(VECTOR_DIM), nullable=False)
    text_ref: Mapped[str] = mapped_column(Text, nullable=False)
    visibility: Mapped[VisibilityLevel] = mapped_column(
        PgEnum(VisibilityLevel, name="visibility_level", create_type=False),  # reuse enum
        default=VisibilityLevel.SELF,
        nullable=False,
    )

    owner: Mapped[User] = relationship(back_populates="embeddings")

    __table_args__ = (
        Index("ix_embeddings_owner_visibility", "owner_id", "visibility"),
    )



#사용자 질문에 대해 실제 데이터 반환 직후 한 줄 남김.
# 철수가 영희의 정보를 조회 시 답을 받으면 , 
# INSERT INTO audit_access(viewer_id, subject_id, detail) VALUES (:철수, :영희, 'events:19:00 미팅 요약');
class AuditAccess(Base):
    __tablename__ = "audit_access"

    access_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    viewer_id: Mapped[int | None] = mapped_column(ForeignKey("users.user_id"))
    subject_id: Mapped[int | None] = mapped_column(ForeignKey("users.user_id"))
    access_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    detail: Mapped[str | None] = mapped_column(Text)

    viewer: Mapped[User] = relationship("User", foreign_keys=[viewer_id])
    subject: Mapped[User] = relationship("User", foreign_keys=[subject_id])


#RAG 검색에서 owner_id IN ~ 조건에 바로 넣어 나와 같은 그룹원들의 데이터만 보도록 필터
def visible_user_ids_subquery(user_id: int):
    """Return a SQLAlchemy selectable yielding IDs visible to *user_id* (self + same group)."""
    from sqlalchemy import select

    gm1 = select(GroupMember.group_id).where(GroupMember.user_id == user_id).subquery()

    return (
        select(GroupMember.user_id)
        .where(GroupMember.group_id.in_(gm1))
        .distinct()
    )


# ---------------------------------------------------------------------------
# EOF
