"""Add execution logs table

Revision ID: add_execution_logs
Revises: add_conversation_tables
Create Date: 2026-02-13

Adds table for:
- execution_logs (monitoring graph execution)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_execution_logs'
down_revision = 'add_conversation_tables'
branch_labels = None
depends_on = None


def upgrade():
    # Create execution_logs table
    op.create_table(
        'execution_logs',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('conversation_id', sa.String(36), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('user_id', sa.String(255), nullable=False, index=True),
        sa.Column('request_message', sa.Text, nullable=False),
        sa.Column('response_message', sa.Text, nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='running', index=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, index=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_duration_ms', sa.Integer, nullable=True),
        sa.Column('execution_trace', postgresql.JSONB, nullable=True),
        sa.Column('node_outputs', postgresql.JSONB, nullable=True),  # Stores node outputs and elapsed time
        sa.Column('error_summary', sa.Text, nullable=True),
        sa.Column('failed_nodes', postgresql.JSONB, nullable=True),
    )

    # Create indexes for common queries
    op.create_index('idx_execution_status_date', 'execution_logs', ['status', 'started_at'])
    op.create_index('idx_execution_user_date', 'execution_logs', ['user_id', 'started_at'])


def downgrade():
    # Drop table
    op.drop_table('execution_logs')
