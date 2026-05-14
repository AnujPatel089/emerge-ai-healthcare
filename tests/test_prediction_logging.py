from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.mlops.prediction_logger import get_prediction_monitoring, log_prediction_monitoring


def test_prediction_monitoring_log_persists():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    row = log_prediction_monitoring(
        db,
        prediction_id=None,
        patient_id="demo",
        model_version="esi_model_v1",
        input_features={"age": 40, "triage_vital_hr": 90},
        predicted_esi="3",
        confidence=0.82,
        safety_rule_triggered=False,
        final_esi="3",
        latency_ms=12.5,
    )

    assert row["model_version"] == "esi_model_v1"
    assert get_prediction_monitoring(db, limit=10)[0]["confidence"] == 0.82
