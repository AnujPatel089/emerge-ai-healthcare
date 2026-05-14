"""add platform incidents

Revision ID: 20260514_0002
Revises: 20260514_0001
Create Date: 2026-05-14
"""

from alembic import op
import sqlalchemy as sa


revision = "20260514_0002"
down_revision = "20260514_0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "platform_incidents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("incident_type", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("related_service", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("resolved_by", sa.String(), nullable=True),
    )
    op.create_index("ix_platform_incidents_incident_type", "platform_incidents", ["incident_type"])
    op.create_index("ix_platform_incidents_severity", "platform_incidents", ["severity"])
    op.create_index("ix_platform_incidents_related_service", "platform_incidents", ["related_service"])
    op.create_index("ix_platform_incidents_status", "platform_incidents", ["status"])
    op.create_index("ix_platform_incidents_created_at", "platform_incidents", ["created_at"])


def downgrade():
    op.drop_table("platform_incidents")
