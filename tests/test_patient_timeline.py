import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.models import PatientTimelineEvent


def test_patient_timeline_event_creation():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    row = PatientTimelineEvent(patient_id="p1", event_type="prediction", event_title="Prediction completed", metadata_json=json.dumps({"source": "test"}))
    db.add(row)
    db.commit()
    assert db.query(PatientTimelineEvent).filter(PatientTimelineEvent.patient_id == "p1").count() == 1
