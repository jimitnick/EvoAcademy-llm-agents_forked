"""
Version history routes:
  GET  /sessions/{session_id}/history
  POST /sessions/{session_id}/rollback
  GET  /sessions/{session_id}/search?q=...
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.version_service import VersionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions/{session_id}")


class RollbackRequest(BaseModel):
    version_number: int = Field(..., description="The version number to rollback to")


# ------------------------------------------------------------------
# GET /sessions/{session_id}/history
# Returns the full version timeline for the session.
# ------------------------------------------------------------------
@router.get("/history")
async def get_history(session_id: str, db: Session = Depends(get_db)):
    """
    Returns the complete version timeline for a session.
    Each entry includes: version_number, operation_type, prompt, summary,
    is_active, file_path, checksum, cells_modified, created_at.
    """
    logger.info(f"[/history] session={session_id}")
    svc = VersionService(db)
    return svc.get_history(session_id)


# ------------------------------------------------------------------
# POST /sessions/{session_id}/rollback
# Metadata-only rollback: changes active_version_id pointer only.
# No files are created, copied, or deleted.
# ------------------------------------------------------------------
@router.post("/rollback")
async def rollback(session_id: str, request: RollbackRequest, db: Session = Depends(get_db)):
    """
    Rolls back to a specific version by updating active_version_id.
    This is a pure metadata operation — no file I/O occurs.
    All versions remain immutable and intact in history.
    """
    logger.info(f"[/rollback] session={session_id} -> version={request.version_number}")
    try:
        svc = VersionService(db)
        result = svc.rollback_to(session_id, request.version_number)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"[/rollback] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# GET /sessions/{session_id}/search?q=...
# Natural language search over version summaries via ChromaDB.
# ------------------------------------------------------------------
@router.get("/search")
async def search_versions(
    session_id: str,
    q: str = Query(..., description="Natural language query, e.g. 'version where tournament selection was added'"),
    n: int = Query(5, description="Max number of results"),
    db: Session = Depends(get_db)
):
    """
    Semantic search over notebook version summaries using ChromaDB.
    Returns the most relevant versions ranked by similarity.
    Example: /sessions/my_session/search?q=elitism was introduced
    """
    logger.info(f"[/search] session={session_id} query='{q}'")
    svc = VersionService(db)
    return svc.semantic_search(session_id, q, n)
