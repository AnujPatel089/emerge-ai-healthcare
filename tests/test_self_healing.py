from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.platform.self_healing import run_checks_and_log, self_healing_status


def test_self_healing_status_returns_contract():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    status = self_healing_status(db, model_loaded=False)
    assert "overall_status" in status
    assert status["model"] == "fallback_active"


def test_run_checks_logs_incidents():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    status = run_checks_and_log(db, model_loaded=False)
    assert status["active_incidents"] >= 1
