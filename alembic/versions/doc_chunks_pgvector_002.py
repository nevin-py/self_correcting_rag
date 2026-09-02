"""pgvector-backed document chunks — replaces embedded ChromaDB.

Revision ID: doc_chunks_pgvector_002
Revises: doc_storage_001
Create Date: 2026-09-02
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "doc_chunks_pgvector_002"
down_revision: Union[str, Sequence[str], None] = "doc_storage_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable the extension (Supabase/dev pgvector images both support this).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "document_chunks",
        sa.Column("id", UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", UUID(), sa.ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("chat_id", UUID(), sa.ForeignKey("chats.chat_id", ondelete="CASCADE"), nullable=True, index=True),
        sa.Column("scope", sa.String(16), nullable=False, server_default="chat"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("metadata", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    # HNSW index for cosine similarity — build AFTER bulk loads.
    op.execute(
        "CREATE INDEX idx_chunks_embedding_hnsw ON document_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX idx_chunks_meta_parent ON document_chunks "
        "USING gin (metadata jsonb_path_ops)"
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
