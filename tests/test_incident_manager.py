from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.platform.incident_manager import create_incident, list_incidents, resolve_incident


def test_incident_lifecycle():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    incident = create_incident(
        db,
        incident_type="prediction_failure",
        severity="critical",
        message="Prediction failed safely.",
        related_service="/predict",
    )
    assert incident["status"] == "open"
    assert len(list_incidents(db)) == 1

    resolved = resolve_incident(db, incident["id"], resolved_by="admin")
    assert resolved["status"] == "resolved"
    assert list_incidents(db) == []
