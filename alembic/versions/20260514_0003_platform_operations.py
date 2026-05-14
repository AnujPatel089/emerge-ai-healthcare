"""platform operations tables

Revision ID: 20260514_0003
Revises: 20260514_0002
Create Date: 2026-05-14
"""

from alembic import op
import sqlalchemy as sa


revision = "20260514_0003"
down_revision = "20260514_0002"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("platform_incidents") as batch:
        batch.add_column(sa.Column("service", sa.String(), nullable=True))
        batch.add_column(sa.Column("detected_at", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("recovery_attempted", sa.Boolean(), nullable=True))
        batch.add_column(sa.Column("recovery_action", sa.String(), nullable=True))
        batch.add_column(sa.Column("recovery_status", sa.String(), nullable=True))
    op.create_index("ix_platform_incidents_service", "platform_incidents", ["service"])
    op.create_index("ix_platform_incidents_detected_at", "platform_incidents", ["detected_at"])
    op.create_index("ix_platform_incidents_recovery_status", "platform_incidents", ["recovery_status"])

    op.create_table(
        "platform_alerts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("alert_type", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("resolved_by", sa.String(), nullable=True),
    )
    op.create_index("ix_platform_alerts_alert_type", "platform_alerts", ["alert_type"])
    op.create_index("ix_platform_alerts_severity", "platform_alerts", ["severity"])
    op.create_index("ix_platform_alerts_status", "platform_alerts", ["status"])
    op.create_index("ix_platform_alerts_created_at", "platform_alerts", ["created_at"])

    op.create_table(
        "patient_timeline_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("patient_id", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("event_title", sa.String(), nullable=False),
        sa.Column("event_description", sa.Text(), nullable=True),
        sa.Column("actor_role", sa.String(), nullable=True),
        sa.Column("actor_name", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
    )
    op.create_index("ix_patient_timeline_events_patient_id", "patient_timeline_events", ["patient_id"])
    op.create_index("ix_patient_timeline_events_event_type", "patient_timeline_events", ["event_type"])
    op.create_index("ix_patient_timeline_events_created_at", "patient_timeline_events", ["created_at"])


def downgrade():
    op.drop_table("patient_timeline_events")
    op.drop_table("platform_alerts")
    with op.batch_alter_table("platform_incidents") as batch:
        batch.drop_column("recovery_status")
        batch.drop_column("recovery_action")
        batch.drop_column("recovery_attempted")
        batch.drop_column("detected_at")
        batch.drop_column("service")
