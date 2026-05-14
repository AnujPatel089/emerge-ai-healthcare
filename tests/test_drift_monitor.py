from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.mlops.drift_monitor import compute_drift_report
from src.models import MLPredictionMonitoring, PredictionLog


def test_drift_monitor_returns_status():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(PredictionLog(age=30, triage_vital_hr=80, triage_vital_rr=16, triage_vital_o2=98, triage_vital_temp=37, triage_vital_sbp=120, triage_vital_dbp=80, cc_chestpain=0, cc_shortnessofbreath=0, final_prediction="ESI 3"))
    db.add(MLPredictionMonitoring(model_version="esi_model_v1", input_features='{"age": 70, "triage_vital_hr": 120, "triage_vital_rr": 24, "triage_vital_o2": 90, "triage_vital_temp": 39, "triage_vital_sbp": 95, "triage_vital_dbp": 60, "cc_chestpain": 1, "cc_shortnessofbreath": 1}', predicted_esi="2", final_esi="2"))
    db.commit()

    report = compute_drift_report(db, "esi_model_v1", persist=False)

    assert report["drift_status"] in {"stable", "warning", "critical"}
    assert "triage_vital_hr" in report["feature_drift"]
