"""initial schema

Revision ID: 4b55fe6e1541
Revises:
Create Date: 2026-08-10 15:17:03.029670

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4b55fe6e1541'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # users
    op.create_table(
        'users',
        sa.Column('user_id', sa.Uuid(), primary_key=True),
        sa.Column('email', sa.String(), nullable=False, unique=True, index=True),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('create_time', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('update_time', sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    # chats
    op.create_table(
        'chats',
        sa.Column('chat_id', sa.Uuid(), primary_key=True),
        sa.Column('user_id', sa.Uuid(), sa.ForeignKey('users.user_id'), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    # agents (interaction log)
    op.create_table(
        'agents',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('chat_id', sa.Uuid(), sa.ForeignKey('chats.chat_id'), nullable=False),
        sa.Column('user_input', sa.String(), nullable=False),
        sa.Column('agent_output', sa.String(), nullable=False),
        sa.Column('routing_path', sa.String(500)),
        sa.Column('token_metric', sa.Integer(), nullable=False),
        sa.Column('latency', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )

    # ingestion_logs
    op.create_table(
        'ingestion_logs',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('chat_id', sa.Uuid(), sa.ForeignKey('chats.chat_id'), nullable=False),
        sa.Column('user_id', sa.Uuid(), sa.ForeignKey('users.user_id'), nullable=False),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('ingestion_logs')
    op.drop_table('agents')
    op.drop_table('chats')
    op.drop_table('users')
