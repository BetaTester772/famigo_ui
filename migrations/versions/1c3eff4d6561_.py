"""empty message

Revision ID: 1c3eff4d6561
Revises: 74ecdaf28434
Create Date: 2025-09-27 13:29:30.003805

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector



# revision identifiers, used by Alembic.
revision: str = '1c3eff4d6561'
down_revision: Union[str, Sequence[str], None] = '74ecdaf28434'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
