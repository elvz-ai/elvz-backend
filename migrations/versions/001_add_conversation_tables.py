"""Add conversation tables

Revision ID: add_conversation_tables
Revises: 
Create Date: 2026-02-11

Adds tables for:
- conversations
- messages  
- query_decompositions
- artifact_batches
- artifacts
- hitl_requests
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_conversation_tables'
down_revision = None  # Update this to the latest migration ID
branch_labels = None
depends_on = None


def upgrade():
    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('user_id', sa.String(36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('thread_id', sa.String(36), unique=True, nullable=False, index=True),
        sa.Column('title', sa.String(255), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='active'),
        sa.Column('metadata', postgresql.JSONB, nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('last_message_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Create indexes for conversations
    op.create_index('idx_conversations_user_status', 'conversations', ['user_id', 'status'])
    op.create_index('idx_conversations_last_message', 'conversations', ['last_message_at'])
    
    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('conversation_id', sa.String(36), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('metadata', postgresql.JSONB, nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    
    op.create_index('idx_messages_conversation_created', 'messages', ['conversation_id', 'created_at'])
    
    # Create query_decompositions table
    op.create_table(
        'query_decompositions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('conversation_id', sa.String(36), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('message_id', sa.String(36), sa.ForeignKey('messages.id', ondelete='SET NULL'), nullable=True),
        sa.Column('original_query', sa.Text, nullable=False),
        sa.Column('is_multi_platform', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('decomposed_queries', postgresql.JSONB, nullable=False, server_default='[]'),
        sa.Column('execution_strategy', sa.String(50), nullable=False, server_default='sequential'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    
    # Create artifact_batches table
    op.create_table(
        'artifact_batches',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('conversation_id', sa.String(36), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('query_decomposition_id', sa.String(36), sa.ForeignKey('query_decompositions.id', ondelete='SET NULL'), nullable=True),
        sa.Column('platforms', postgresql.ARRAY(sa.String), nullable=False),
        sa.Column('topic', sa.String(500), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='pending'),
        sa.Column('execution_strategy', sa.String(50), nullable=False, server_default='sequential'),
        sa.Column('total_tokens_used', sa.Integer, nullable=False, server_default='0'),
        sa.Column('total_cost', sa.Float, nullable=False, server_default='0.0'),
        sa.Column('execution_time_ms', sa.Integer, nullable=False, server_default='0'),
        sa.Column('metadata', postgresql.JSONB, nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Create artifacts table
    op.create_table(
        'artifacts',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('conversation_id', sa.String(36), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('message_id', sa.String(36), sa.ForeignKey('messages.id', ondelete='SET NULL'), nullable=True),
        sa.Column('batch_id', sa.String(36), sa.ForeignKey('artifact_batches.id', ondelete='SET NULL'), nullable=True, index=True),
        sa.Column('artifact_type', sa.String(50), nullable=False),
        sa.Column('platform', sa.String(50), nullable=True, index=True),
        sa.Column('content', postgresql.JSONB, nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='draft'),
        sa.Column('user_rating', sa.Integer, nullable=True),
        sa.Column('user_feedback', sa.Text, nullable=True),
        sa.Column('was_edited', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('was_published', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('generation_metadata', postgresql.JSONB, nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    op.create_index('idx_artifacts_type_platform', 'artifacts', ['artifact_type', 'platform'])
    op.create_index('idx_artifacts_status', 'artifacts', ['status'])
    op.create_index('idx_artifacts_created', 'artifacts', ['created_at'])
    
    # Create hitl_requests table
    op.create_table(
        'hitl_requests',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('conversation_id', sa.String(36), sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('artifact_id', sa.String(36), sa.ForeignKey('artifacts.id', ondelete='SET NULL'), nullable=True),
        sa.Column('request_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='pending'),
        sa.Column('prompt', sa.Text, nullable=False),
        sa.Column('options', postgresql.JSONB, nullable=True),
        sa.Column('response', sa.Text, nullable=True),
        sa.Column('selected_options', postgresql.JSONB, nullable=True),
        sa.Column('context', postgresql.JSONB, nullable=False, server_default='{}'),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('requested_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('responded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('requester_notes', sa.Text, nullable=True),
        sa.Column('responder_notes', sa.Text, nullable=True),
    )
    
    op.create_index('idx_hitl_status', 'hitl_requests', ['status'])
    op.create_index('idx_hitl_conversation_status', 'hitl_requests', ['conversation_id', 'status'])
    op.create_index('idx_hitl_expires', 'hitl_requests', ['expires_at'])


def downgrade():
    # Drop tables in reverse order
    op.drop_table('hitl_requests')
    op.drop_table('artifacts')
    op.drop_table('artifact_batches')
    op.drop_table('query_decompositions')
    op.drop_table('messages')
    op.drop_table('conversations')
