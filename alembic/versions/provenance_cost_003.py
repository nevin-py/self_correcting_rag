"""Persist provenance + cost columns for message analysis restore.

Revision ID: provenance_cost_003
Revises: refresh_tokens_002
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "provenance_cost_003"
down_revision: Union[str, Sequence[str], None] = "refresh_tokens_002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("agents", sa.Column("provenance_json", sa.Text(), nullable=True))
    op.add_column("agents", sa.Column("estimated_cost_usd", sa.Float(), nullable=True))
    op.add_column("agents", sa.Column("provider_used", sa.String(length=32), nullable=True))
    op.add_column("chat_messages", sa.Column("provenance_json", sa.Text(), nullable=True))
    op.add_column("chat_messages", sa.Column("token_estimate", sa.Integer(), nullable=True))
    op.add_column("chat_messages", sa.Column("estimated_cost_usd", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_messages", "estimated_cost_usd")
    op.drop_column("chat_messages", "token_estimate")
    op.drop_column("chat_messages", "provenance_json")
    op.drop_column("agents", "provider_used")
    op.drop_column("agents", "estimated_cost_usd")
    op.drop_column("agents", "provenance_json")
