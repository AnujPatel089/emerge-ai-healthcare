from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.mlops.model_registry import ModelRegistryRecord, get_current_production_model, promote_model, register_model


def _db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)
    return session()


def test_register_and_promote_model():
    db = _db()
    first = register_model(
        db,
        ModelRegistryRecord(
            model_name="esi_model",
            model_version="esi_model_v1",
            model_path="models/registry/esi_model_v1.pkl",
            status="staging",
            f1_score=0.7,
        ),
    )
    assert first["model_version"] == "esi_model_v1"

    promoted = promote_model(db, "esi_model_v1", deployed_by="admin")
    assert promoted["status"] == "production"
    assert get_current_production_model(db)["model_version"] == "esi_model_v1"
