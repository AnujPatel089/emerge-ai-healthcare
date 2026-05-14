from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------

load_dotenv()

# -----------------------------
# DATABASE URL
# -----------------------------

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured. Add it to .env for local development "
        "or set it in your deployment environment."
    )

# -----------------------------
# SQLALCHEMY ENGINE
# -----------------------------

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=True
)

# -----------------------------
# SESSION
# -----------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# -----------------------------
# BASE MODEL
# -----------------------------

Base = declarative_base()

# -----------------------------
# DB DEPENDENCY
# -----------------------------

def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()
