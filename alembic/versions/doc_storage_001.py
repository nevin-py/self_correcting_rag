"""Persist uploaded document originals so citations can hyperlink to sources.

Revision ID: doc_storage_001
Revises: provider_custom_baseurl_001
Create Date: 2026-09-02
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "doc_storage_001"
down_revision: Union[str, Sequence[str], None] = "provider_custom_baseurl_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("ingestion_logs", sa.Column("storage_path", sa.String(512), nullable=True))
    op.add_column("ingestion_logs", sa.Column("content_type", sa.String(128), nullable=True))
    op.add_column("ingestion_logs", sa.Column("size_bytes", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingestion_logs", "size_bytes")
    op.drop_column("ingestion_logs", "content_type")
    op.drop_column("ingestion_logs", "storage_path")
