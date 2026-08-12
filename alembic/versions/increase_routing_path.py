"""Increase routing_path column size

Revision ID: increase_routing_path
Revises: abc123
Create Date: 2024-01-01

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'increase_routing_path'
down_revision: Union[str, None] = 'abc123'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('agents', 'routing_path',
                   existing_type=sa.String(500),
                   type_=sa.String(50000),
                   existing_nullable=True)


def downgrade() -> None:
    op.alter_column('agents', 'routing_path',
                   existing_type=sa.String(50000),
                   type_=sa.String(500),
                   existing_nullable=True)
