from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.platform.incident_manager import create_incident


def test_incident_has_recovery_fields():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    incident = create_incident(db, incident_type="model_failure", severity="critical", message="model failed", related_service="model")
    assert incident["recovery_status"] == "detected"
    assert incident["recovery_attempted"] is False
