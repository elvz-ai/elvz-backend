"""Add user_style_profiles table

Revision ID: add_user_style_profiles
Revises: add_node_outputs_column
Create Date: 2026-02-23

Stores pre-computed writing style features per user (avg word count, emoji usage,
hook style, etc.). Written at webhook time, read at chat time to avoid re-scanning
Qdrant on every request.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_user_style_profiles'
down_revision = 'add_node_outputs_column'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'user_style_profiles',
        sa.Column('user_id', sa.String(64), primary_key=True),
        sa.Column('features', postgresql.JSONB, nullable=False),
        sa.Column('posts_analyzed', sa.Integer, nullable=False, server_default='0'),
        sa.Column('confidence', sa.String(20), nullable=False, server_default='VERY LOW'),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('NOW()'),
        ),
    )


def downgrade():
    op.drop_table('user_style_profiles')
