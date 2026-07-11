"""
NotebookRepository — CRUD for Notebook and Session.
Manages the active_version_id pointer (the only field that changes on rollback).
"""
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session as DBSession

from app.db.models import Notebook, Session as SessionModel

logger = logging.getLogger(__name__)


class NotebookRepository:
    def __init__(self, db: DBSession):
        self.db = db

    def get_or_create_session(self, session_id: str) -> SessionModel:
        """Ensure a Session row exists for this session_id."""
        session = self.db.get(SessionModel, session_id)
        if not session:
            session = SessionModel(session_id=session_id, created_at=datetime.utcnow())
            self.db.add(session)
            self.db.flush()
            logger.info(f"[DB] Created new session: {session_id}")
        else:
            session.last_active_at = datetime.utcnow()
        return session

    def get_or_create_notebook(self, session_id: str) -> Notebook:
        """
        Returns the Notebook for a session, creating one if it doesn't exist.
        Each session has exactly one Notebook.
        """
        self.get_or_create_session(session_id)

        notebook = (
            self.db.query(Notebook)
            .filter(Notebook.session_id == session_id)
            .first()
        )
        if not notebook:
            notebook = Notebook(session_id=session_id)
            self.db.add(notebook)
            self.db.flush()
            logger.info(f"[DB] Created notebook for session: {session_id}")
        return notebook

    def get_notebook_by_session(self, session_id: str) -> Optional[Notebook]:
        return (
            self.db.query(Notebook)
            .filter(Notebook.session_id == session_id)
            .first()
        )

    def set_active_version(self, notebook_id: str, version_id: str) -> None:
        """
        The ONLY write path for rollback — updates active_version_id pointer.
        No file I/O, no notebook reconstruction.
        """
        notebook = self.db.get(Notebook, notebook_id)
        if notebook:
            notebook.active_version_id = version_id
            notebook.updated_at = datetime.utcnow()
            self.db.flush()
            logger.info(f"[DB] Set active version to {version_id} for notebook {notebook_id}")

    def clear_session_notebooks(self, session_id: str) -> None:
        """
        Called on /generate to reset a session's history (new problem).
        Deletion order must be explicit to avoid SQLAlchemy circular-FK errors:
          1. NULL out active_version_id (breaks Notebook ↔ NotebookVersion cycle)
          2. Delete VersionOperation rows (deepest dependency)
          3. Delete NotebookVersion rows
          4. Delete Notebook rows
        """
        from app.db.models import NotebookVersion, VersionOperation

        notebooks = (
            self.db.query(Notebook)
            .filter(Notebook.session_id == session_id)
            .all()
        )
        if not notebooks:
            return

        notebook_ids = [nb.notebook_id for nb in notebooks]

        # Step 1: break the circular FK — null out active_version_id pointers
        for nb in notebooks:
            nb.active_version_id = None
        self.db.flush()

        # Step 2: collect all version IDs for this session's notebooks
        versions = (
            self.db.query(NotebookVersion)
            .filter(NotebookVersion.notebook_id.in_(notebook_ids))
            .all()
        )
        version_ids = [v.version_id for v in versions]

        # Step 3: delete audit log entries (deepest dependency)
        if version_ids:
            self.db.query(VersionOperation).filter(
                VersionOperation.version_id.in_(version_ids)
            ).delete(synchronize_session=False)

        # Step 4: delete version rows (null parent_version_id self-refs first)
        for v in versions:
            v.parent_version_id = None
        self.db.flush()
        for v in versions:
            self.db.delete(v)
        self.db.flush()

        # Step 5: delete notebook rows
        for nb in notebooks:
            self.db.delete(nb)
        self.db.flush()

        logger.info(f"[DB] Cleared session history for: {session_id} ({len(versions)} versions removed)")
