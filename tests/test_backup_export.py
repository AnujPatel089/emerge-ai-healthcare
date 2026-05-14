from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base
from src.models import AppUser
from src.platform.backup_manager import export_backup


def test_backup_export_excludes_secrets_and_hashes():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(AppUser(full_name="A", username="a", email="a@example.com", password_hash="secret_hash", role="admin", requested_role="admin"))
    db.commit()
    data = export_backup(db)
    text = str(data["users_metadata"])
    assert "secret_hash" not in text
    assert "DATABASE_URL" not in text
    assert "DATABASE_URL" in data["metadata"]["excluded"]
