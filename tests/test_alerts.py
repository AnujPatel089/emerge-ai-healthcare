from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.platform.alert_manager import create_alert, list_alerts, resolve_alert


def test_alert_creation_resolution():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    alert = create_alert(db, "model_failure", "critical", "Model failure", "Fallback active")
    assert len(list_alerts(db)) == 1
    resolved = resolve_alert(db, alert["id"], "admin")
    assert resolved["status"] == "resolved"
