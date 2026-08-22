"""LLM call tracing table for observability.

Revision ID: llm_traces_004
Revises: provenance_cost_003
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import uuid  # noqa: F401


revision: str = "llm_traces_004"
down_revision: Union[str, Sequence[str], None] = "provenance_cost_003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_call_traces",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
        sa.Column("chat_id", sa.UUID(), nullable=True, index=True),
        sa.Column("user_id", sa.UUID(), nullable=True, index=True),
        sa.Column("role", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("node", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("provider", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("model", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="ok"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("prompt_chars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_chars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prompt_tokens_est", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens_est", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_llm_traces_role_model", "llm_call_traces", ["role", "model"])


def downgrade() -> None:
    op.drop_index("ix_llm_traces_role_model", table_name="llm_call_traces")
    op.drop_table("llm_call_traces")
