"""Add node_outputs column to execution_logs

Revision ID: add_node_outputs_column
Revises: add_execution_logs
Create Date: 2026-02-13

Adds node_outputs column to execution_logs table
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_node_outputs_column'
down_revision = 'add_execution_logs'
branch_labels = None
depends_on = None


def upgrade():
    # Add node_outputs column to existing execution_logs table
    op.add_column(
        'execution_logs',
        sa.Column('node_outputs', postgresql.JSONB, nullable=True)
    )


def downgrade():
    # Remove node_outputs column
    op.drop_column('execution_logs', 'node_outputs')
