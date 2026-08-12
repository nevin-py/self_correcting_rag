"""Add file_hash column to ingestion_logs for content-based deduplication

Revision ID: abc123
Revises: 8065a711398e
Create Date: 2026-08-10 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'abc123'
down_revision: Union[str, Sequence[str], None] = '8065a711398e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add file_hash column for content-based deduplication."""
    op.add_column(
        'ingestion_logs',
        sa.Column('file_hash', sa.String(64), nullable=True)
    )
    # Create index for faster lookups
    op.create_index(
        'ix_ingestion_logs_file_hash',
        'ingestion_logs',
        ['file_hash'],
        unique=False
    )
    op.create_index(
        'ix_ingestion_logs_user_id_file_hash',
        'ingestion_logs',
        ['user_id', 'file_hash'],
        unique=False
    )


def downgrade() -> None:
    """Remove file_hash column."""
    op.drop_index('ix_ingestion_logs_user_id_file_hash', table_name='ingestion_logs')
    op.drop_index('ix_ingestion_logs_file_hash', table_name='ingestion_logs')
    op.drop_column('ingestion_logs', 'file_hash')
