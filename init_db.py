from src.database import engine
from src.models import Base

print("Creating database tables...")

Base.metadata.create_all(bind=engine)

print("Database tables created successfully!")