"""
VersionRepository — CRUD for NotebookVersion and VersionOperation.
"""
import logging
from typing import List, Optional

from sqlalchemy.orm import Session as DBSession

from app.db.models import NotebookVersion, VersionOperation

logger = logging.getLogger(__name__)


class VersionRepository:
    def __init__(self, db: DBSession):
        self.db = db

    def get_next_version_number(self, notebook_id: str) -> int:
        from sqlalchemy import func
        result = (
            self.db.query(func.max(NotebookVersion.version_number))
            .filter(NotebookVersion.notebook_id == notebook_id)
            .scalar()
        )
        return (result or 0) + 1

    def create_version(
        self,
        notebook_id: str,
        version_number: int,
        operation_type: str,
        file_path: str,
        checksum: str,
        is_snapshot: bool = True,
        delta_size: Optional[int] = None,
        prompt: Optional[str] = None,
        summary: Optional[str] = None,
        parent_version_id: Optional[str] = None,
        cells_modified: Optional[List[str]] = None,
        extra_metadata: Optional[dict] = None,
    ) -> NotebookVersion:
        version = NotebookVersion(
            notebook_id=notebook_id,
            version_number=version_number,
            parent_version_id=parent_version_id,
            operation_type=operation_type,
            prompt=prompt,
            summary=summary,
            file_path=file_path,
            checksum=checksum,
            is_snapshot=is_snapshot,
            delta_size=delta_size,
            cells_modified=cells_modified or [],
            extra_metadata=extra_metadata or {}
        )
        self.db.add(version)
        self.db.flush()
        logger.info(
            f"[DB] Created version v{version_number} ({operation_type}) "
            f"{'[snapshot]' if is_snapshot else '[delta]'} for notebook {notebook_id}"
        )
        return version

    def get_version_by_id(self, version_id: str) -> Optional[NotebookVersion]:
        return self.db.get(NotebookVersion, version_id)

    def get_version_by_number(self, notebook_id: str, version_number: int) -> Optional[NotebookVersion]:
        return (
            self.db.query(NotebookVersion)
            .filter(
                NotebookVersion.notebook_id == notebook_id,
                NotebookVersion.version_number == version_number
            )
            .first()
        )

    def get_history(self, notebook_id: str) -> List[NotebookVersion]:
        """Returns all versions ordered by version_number ascending."""
        return (
            self.db.query(NotebookVersion)
            .filter(NotebookVersion.notebook_id == notebook_id)
            .order_by(NotebookVersion.version_number.asc())
            .all()
        )

    def mark_chroma_indexed(self, version_id: str) -> None:
        version = self.db.get(NotebookVersion, version_id)
        if version:
            version.chroma_indexed = True
            self.db.flush()

    def log_operation(self, version_id: str, action: str, details: str = "") -> None:
        op = VersionOperation(version_id=version_id, action=action, details=details)
        self.db.add(op)
        self.db.flush()
        logger.info(f"[AuditLog] {action} on version {version_id}: {details}")
