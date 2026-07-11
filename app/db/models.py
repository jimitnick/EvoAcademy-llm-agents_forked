"""
SQLAlchemy ORM models.
Tables: Session, Notebook, NotebookVersion, VersionOperation
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, ForeignKey, Integer, String, Text, JSON
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    notebooks: Mapped[list["Notebook"]] = relationship(
        "Notebook", back_populates="session", cascade="all, delete-orphan"
    )


class Notebook(Base):
    __tablename__ = "notebooks"

    notebook_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), nullable=False)
    # Nullable: no version exists until /generate is called
    active_version_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("notebook_versions.version_id", use_alter=True, name="fk_active_version"),
        nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    session: Mapped["Session"] = relationship("Session", back_populates="notebooks")
    versions: Mapped[list["NotebookVersion"]] = relationship(
        "NotebookVersion",
        primaryjoin="Notebook.notebook_id == NotebookVersion.notebook_id",
        back_populates="notebook",
        cascade="all, delete-orphan",
        foreign_keys="NotebookVersion.notebook_id"
    )
    active_version: Mapped[Optional["NotebookVersion"]] = relationship(
        "NotebookVersion",
        foreign_keys=[active_version_id],
        primaryjoin="Notebook.active_version_id == NotebookVersion.version_id",
        post_update=True,   # breaks the circular FK cycle on INSERT and DELETE
    )


class NotebookVersion(Base):
    __tablename__ = "notebook_versions"

    version_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    notebook_id: Mapped[str] = mapped_column(
        String, ForeignKey("notebooks.notebook_id"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_version_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("notebook_versions.version_id"), nullable=True
    )
    operation_type: Mapped[str] = mapped_column(String, nullable=False)
    # generate | refine | debug | rollback
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    chroma_indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    cells_modified: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    notebook: Mapped["Notebook"] = relationship(
        "Notebook",
        foreign_keys=[notebook_id],
        back_populates="versions"
    )
    parent: Mapped[Optional["NotebookVersion"]] = relationship(
        "NotebookVersion", remote_side="NotebookVersion.version_id",
        foreign_keys=[parent_version_id]
    )
    operations: Mapped[list["VersionOperation"]] = relationship(
        "VersionOperation", back_populates="version", cascade="all, delete-orphan"
    )

    def to_dict(self, is_active: bool = False) -> dict:
        return {
            "version_id": self.version_id,
            "version_number": self.version_number,
            "parent_version_id": self.parent_version_id,
            "operation_type": self.operation_type,
            "prompt": self.prompt,
            "summary": self.summary,
            "file_path": self.file_path,
            "checksum": self.checksum,
            "cells_modified": self.cells_modified or [],
            "is_active": is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "extra_metadata": self.extra_metadata or {}
        }


class VersionOperation(Base):
    """Audit log for version lifecycle events."""
    __tablename__ = "version_operations"

    op_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    version_id: Mapped[str] = mapped_column(
        String, ForeignKey("notebook_versions.version_id"), nullable=False
    )
    action: Mapped[str] = mapped_column(String, nullable=False)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    version: Mapped["NotebookVersion"] = relationship("NotebookVersion", back_populates="operations")
