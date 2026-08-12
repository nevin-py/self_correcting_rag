"""Change routing_path to TEXT type (unlimited)

Revision ID: routing_path_text
Revises: increase_routing_path
Create Date: 2024-01-01

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'routing_path_text'
down_revision: Union[str, None] = 'increase_routing_path'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Change from VARCHAR(50000) to TEXT (unlimited size)
    op.alter_column('agents', 'routing_path',
                   existing_type=sa.String(50000),
                   type_=sa.Text,
                   existing_nullable=True)


def downgrade() -> None:
    op.alter_column('agents', 'routing_path',
                   existing_type=sa.Text,
                   type_=sa.String(50000),
                   existing_nullable=True)
