"""Add email verification, OTP, provider settings, usage events, ingest tokens.

Revision ID: auth_keys_memory_001
Revises: abc123
Create Date: 2026-08-12
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "auth_keys_memory_001"
down_revision: Union[str, Sequence[str], None] = "routing_path_text"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("email_verified", sa.Boolean(), server_default="false", nullable=False),
    )
    op.add_column(
        "users",
        sa.Column("email_verified_at", sa.DateTime(), nullable=True),
    )
    # Existing accounts remain usable
    op.execute("UPDATE users SET email_verified = true WHERE email_verified = false")

    op.create_table(
        "email_otps",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("code_hash", sa.String(length=128), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("consumed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_email_otps_user_id", "email_otps", ["user_id"])

    op.create_table(
        "user_provider_settings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("api_key_enc", sa.Text(), nullable=True),
        sa.Column("fallback_api_key_enc", sa.Text(), nullable=True),
        sa.Column("masked_key", sa.String(length=64), nullable=True),
        sa.Column("masked_fallback_key", sa.String(length=64), nullable=True),
        sa.Column("planner_model", sa.String(length=128), nullable=True),
        sa.Column("generator_model", sa.String(length=128), nullable=True),
        sa.Column("verifier_model", sa.String(length=128), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "provider", name="uq_user_provider"),
    )
    op.create_index("ix_user_provider_settings_user_id", "user_provider_settings", ["user_id"])

    op.create_table(
        "usage_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("amount", sa.Integer(), server_default="1", nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_usage_events_user_id", "usage_events", ["user_id"])
    op.create_index("ix_usage_events_kind", "usage_events", ["kind"])
    op.create_index("ix_usage_events_created_at", "usage_events", ["created_at"])

    op.add_column(
        "ingestion_logs",
        sa.Column("ingest_token_count", sa.Integer(), server_default="0", nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ingestion_logs", "ingest_token_count")
    op.drop_index("ix_usage_events_created_at", table_name="usage_events")
    op.drop_index("ix_usage_events_kind", table_name="usage_events")
    op.drop_index("ix_usage_events_user_id", table_name="usage_events")
    op.drop_table("usage_events")
    op.drop_index("ix_user_provider_settings_user_id", table_name="user_provider_settings")
    op.drop_table("user_provider_settings")
    op.drop_index("ix_email_otps_user_id", table_name="email_otps")
    op.drop_table("email_otps")
    op.drop_column("users", "email_verified_at")
    op.drop_column("users", "email_verified")
