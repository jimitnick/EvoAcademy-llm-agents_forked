"""POST /debug — Auto-fix a runtime traceback in the active notebook."""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.version_service import VersionService

logger = logging.getLogger(__name__)
router = APIRouter()


class DebugRequest(BaseModel):
    session_id: str
    traceback_msg: str = Field(..., description="The runtime error thrown by the Jupyter kernel")
    current_cells: Dict[str, str] = Field(..., description="The current state of the 12 DEAP cells")


class DebugResponse(BaseModel):
    status: str
    cells: Dict[str, str]
    cells_modified: List[str]
    tutor_explanation: str
    version_number: int
    version_id: str


@router.post("/debug", response_model=DebugResponse)
async def debug_notebook(request: DebugRequest, db: Session = Depends(get_db)):
    """
    Auto-fix a runtime error in the active notebook.
    Creates a new immutable version with the fixed code.
    If the fix fails validation, returns a 500 (no new version created).
    """
    logger.info(f"[/debug] session={request.session_id}")
    try:
        svc = VersionService(db)
        result = await svc.debug(
            session_id=request.session_id,
            traceback_msg=request.traceback_msg,
            current_cells=request.current_cells,
        )
        return DebugResponse(**result)
    except Exception as e:
        logger.exception(f"[/debug] Failed: {e}")
        try:
            svc = VersionService(db)
            active_cells = svc.get_active_cells(request.session_id) or {}
            
            notebook = svc.notebook_repo.get_notebook_by_session(request.session_id)
            active_ver_num = 1
            active_ver_id = ""
            if notebook and notebook.active_version_id:
                active_ver = svc.version_repo.get_version_by_id(notebook.active_version_id)
                if active_ver:
                    active_ver_num = active_ver.version_number
                    active_ver_id = active_ver.version_id

            return DebugResponse(
                status="reverted",
                cells=active_cells,
                cells_modified=[],
                tutor_explanation=f"Debug pipeline error: {str(e)}. Automatically rolled back to the previous working version.",
                version_number=active_ver_num,
                version_id=active_ver_id,
            )
        except Exception as fallback_err:
            logger.exception(f"[/debug] Rollback fallback failed: {fallback_err}")
            raise HTTPException(status_code=500, detail=str(e))
