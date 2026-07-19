"""
SQLAlchemy database engine and session factory.
Defaults to SQLite. Set DATABASE_URL env var to switch to PostgreSQL.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./evo_academy.db")

# SQLite needs check_same_thread=False for FastAPI async usage
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def _migrate_add_delta_columns():
    """
    Idempotent migration: add delta-compression columns to notebook_versions
    if they don't already exist.  SQLite does not support IF NOT EXISTS on
    ALTER TABLE, so we catch the OperationalError.
    """
    with engine.connect() as conn:
        for col_def in [
            "ALTER TABLE notebook_versions ADD COLUMN is_snapshot BOOLEAN NOT NULL DEFAULT 1",
            "ALTER TABLE notebook_versions ADD COLUMN delta_size INTEGER",
        ]:
            try:
                conn.execute(__import__("sqlalchemy").text(col_def))
                conn.commit()
            except Exception:
                pass  # Column already exists — safe to ignore


def init_db():
    """Create all tables and apply lightweight migrations. Called on app startup."""
    from app.db import models  # noqa: F401 — import models so Base knows about them
    Base.metadata.create_all(bind=engine)
    _migrate_add_delta_columns()


def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
