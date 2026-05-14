"""add mlops tables

Revision ID: 20260514_0001
Revises:
Create Date: 2026-05-14
"""

from alembic import op
import sqlalchemy as sa


revision = "20260514_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ml_model_registry",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("model_path", sa.String(), nullable=False),
        sa.Column("feature_columns_path", sa.String(), nullable=True),
        sa.Column("training_dataset", sa.String(), nullable=True),
        sa.Column("training_date", sa.DateTime(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1_score", sa.Float(), nullable=True),
        sa.Column("recall_esi_1_2", sa.Float(), nullable=True),
        sa.Column("confusion_matrix_path", sa.String(), nullable=True),
        sa.Column("feature_importance_path", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("deployed_at", sa.DateTime(), nullable=True),
        sa.Column("deployed_by", sa.String(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_ml_model_registry_model_name", "ml_model_registry", ["model_name"])
    op.create_index("ix_ml_model_registry_model_version", "ml_model_registry", ["model_version"], unique=True)
    op.create_index("ix_ml_model_registry_status", "ml_model_registry", ["status"])

    op.create_table(
        "ml_prediction_monitoring",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("prediction_id", sa.Integer(), nullable=True),
        sa.Column("patient_id", sa.String(), nullable=True),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("input_features", sa.Text(), nullable=False),
        sa.Column("predicted_esi", sa.String(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("icu_risk", sa.Float(), nullable=True),
        sa.Column("readmission_risk", sa.Float(), nullable=True),
        sa.Column("safety_rule_triggered", sa.Boolean(), nullable=True),
        sa.Column("doctor_override", sa.Boolean(), nullable=True),
        sa.Column("final_esi", sa.String(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("failed", sa.Boolean(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["prediction_id"], ["prediction_logs.id"]),
    )
    op.create_index("ix_ml_prediction_monitoring_prediction_id", "ml_prediction_monitoring", ["prediction_id"])
    op.create_index("ix_ml_prediction_monitoring_patient_id", "ml_prediction_monitoring", ["patient_id"])
    op.create_index("ix_ml_prediction_monitoring_model_version", "ml_prediction_monitoring", ["model_version"])
    op.create_index("ix_ml_prediction_monitoring_timestamp", "ml_prediction_monitoring", ["timestamp"])

    op.create_table(
        "ml_drift_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("drift_score", sa.Float(), nullable=False),
        sa.Column("drift_status", sa.String(), nullable=False),
        sa.Column("feature_drift", sa.Text(), nullable=False),
        sa.Column("baseline_window", sa.String(), nullable=True),
        sa.Column("live_window", sa.String(), nullable=True),
        sa.Column("recommendation", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_ml_drift_reports_model_version", "ml_drift_reports", ["model_version"])
    op.create_index("ix_ml_drift_reports_drift_status", "ml_drift_reports", ["drift_status"])
    op.create_index("ix_ml_drift_reports_created_at", "ml_drift_reports", ["created_at"])

    op.create_table(
        "ml_model_cards",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("card_markdown", sa.Text(), nullable=False),
        sa.Column("card_path", sa.String(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_ml_model_cards_model_version", "ml_model_cards", ["model_version"], unique=True)
    op.create_index("ix_ml_model_cards_created_at", "ml_model_cards", ["created_at"])


def downgrade():
    op.drop_table("ml_model_cards")
    op.drop_table("ml_drift_reports")
    op.drop_table("ml_prediction_monitoring")
    op.drop_table("ml_model_registry")
