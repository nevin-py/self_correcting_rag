"""add provider base_url + client_family

Revision ID: provider_custom_baseurl_001
Revises: increase_routing_path
Create Date: 2026-08-27

Supports arbitrary provider keys: store a custom OpenAI-compatible base URL and
a client-family hint (openai | anthropic | ollama) alongside each provider row.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "provider_custom_baseurl_001"
down_revision = "llm_traces_004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("user_provider_settings") as batch:
        batch.add_column(sa.Column("client_family", sa.String(16), nullable=False, server_default="openai"))
        batch.add_column(sa.Column("base_url", sa.String(255), nullable=True))
    # widen provider from 32 to 64 chars for longer custom ids
    with op.batch_alter_table("user_provider_settings") as batch:
        batch.alter_column("provider", existing_type=sa.String(32), type_=sa.String(64), existing_nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("user_provider_settings") as batch:
        batch.drop_column("base_url")
        batch.drop_column("client_family")
    with op.batch_alter_table("user_provider_settings") as batch:
        batch.alter_column("provider", existing_type=sa.String(64), type_=sa.String(32), existing_nullable=False)