from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.mlops.model_registry import ModelRegistryRecord, register_model
from src.models import MLPredictionMonitoring
from src.platform.ai_reliability import ai_reliability_status


def test_ai_reliability_detects_possible_under_triage():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    register_model(
        db,
        ModelRegistryRecord(
            model_name="esi_model",
            model_version="esi_model_v1",
            model_path="models/registry/esi_model_v1.pkl",
            status="production",
        ),
    )
    db.add(
        MLPredictionMonitoring(
            model_version="esi_model_v1",
            input_features='{"age": 80}',
            predicted_esi="4",
            final_esi="4",
            confidence=0.4,
            icu_risk=0.9,
            safety_rule_triggered=True,
        )
    )
    db.commit()

    result = ai_reliability_status(db)

    assert result["status"] == "critical"
    assert result["possible_under_triage_patterns"] == 1
    assert "clinician review required" in result["message"].lower()
