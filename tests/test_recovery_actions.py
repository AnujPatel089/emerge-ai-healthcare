from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.platform.recovery_actions import reconnect_database, safe_recovery_for_incident


def test_database_reconnect_rolls_back_safely():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    result = reconnect_database(db)
    assert result["status"] == "recovered"


def test_model_failure_activates_fallback():
    result = safe_recovery_for_incident("model_failure")
    assert result["status"] == "recovered"
    assert result["action"] == "rule_based_triage_fallback"
