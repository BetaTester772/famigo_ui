from __future__ import annotations

"""
Initial schema: extensions, enum, tables, indexes (incl. pgvector ivfflat cosine).

Revision ID: 20250826_000001
Revises    : None
Create Date: 2025-08-26
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# pgvector helper (fallback if package missing)
try:  # pragma: no cover
    from pgvector.sqlalchemy import Vector  # type: ignore
except Exception:  # pragma: no cover
    class Vector(sa.types.UserDefinedType):
        def get_col_spec(self, **kwargs):
            return "vector(1536)"

revision = "20250826_000001"
down_revision = None
branch_labels = None
depends_on = None

VECTOR_DIM = 1536


def upgrade() -> None:
    # Extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Enum type
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE visibility_level AS ENUM ('self','group','public');
        EXCEPTION WHEN duplicate_object THEN NULL;
        END $$;
        """
    )

    # users
    op.create_table(
        "users",
        sa.Column("user_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("face_id", postgresql.UUID(as_uuid=True), nullable=False,
                  server_default=sa.text("gen_random_uuid()"), unique=True),
        sa.Column("name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("profile_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )

    # groups
    op.create_table(
        "groups",
        sa.Column("group_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )

    # group_members (association)
    op.create_table(
        "group_members",
        sa.Column("group_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False, server_default="member"),
        sa.Column("joined_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("group_id", "user_id"),
        sa.ForeignKeyConstraint(["group_id"], ["groups.group_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_group_members_user_id", "group_members", ["user_id"], unique=False)

    # events
    op.create_table(
        "events",
        sa.Column("event_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("location", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("visibility", sa.Enum("self", "group", "public", name="visibility_level"),
                  nullable=False, server_default="self"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["owner_id"], ["users.user_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_events_owner_visibility", "events", ["owner_id", "visibility"], unique=False)

    # embeddings
    op.create_table(
        "embeddings",
        sa.Column("embedding_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("vector", Vector(VECTOR_DIM), nullable=False),
        sa.Column("text_ref", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("visibility", sa.Enum("self", "group", "public", name="visibility_level"),
                  nullable=False, server_default="self"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["owner_id"], ["users.user_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_embeddings_owner_visibility", "embeddings", ["owner_id", "visibility"], unique=False)
    op.create_index("ix_embeddings_text_ref", "embeddings", ["text_ref"], unique=False)

    # ivfflat index for cosine distance
    op.execute(
        """
        DO $$ BEGIN
            CREATE INDEX IF NOT EXISTS ix_embeddings_vector_ivfflat_cosine
            ON embeddings USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);
        EXCEPTION WHEN undefined_object THEN
            -- older pgvector versions may not expose vector_cosine_ops
            CREATE INDEX ix_embeddings_vector_ivfflat_cosine
            ON embeddings USING ivfflat (vector) WITH (lists = 100);
        END $$;
        """
    )
    op.execute("ANALYZE embeddings;")

    # audit_access
    op.create_table(
        "audit_access",
        sa.Column("access_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("viewer_id", sa.Integer(), nullable=True),
        sa.Column("subject_id", sa.Integer(), nullable=True),
        sa.Column("access_time", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["viewer_id"], ["users.user_id"]),
        sa.ForeignKeyConstraint(["subject_id"], ["users.user_id"]),
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_embeddings_vector_ivfflat_cosine;")

    op.drop_table("audit_access")
    op.drop_index("ix_embeddings_text_ref", table_name="embeddings")
    op.drop_index("ix_embeddings_owner_visibility", table_name="embeddings")
    op.drop_table("embeddings")
    op.drop_index("ix_events_owner_visibility", table_name="events")
    op.drop_table("events")
    op.drop_index("ix_group_members_user_id", table_name="group_members")
    op.drop_table("group_members")
    op.drop_table("groups")
    op.drop_table("users")

    op.execute("DROP TYPE IF EXISTS visibility_level;")
