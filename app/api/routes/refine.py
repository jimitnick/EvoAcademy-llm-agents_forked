"""POST /refine — Refine existing notebook based on a user request."""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.version_service import VersionService

logger = logging.getLogger(__name__)
router = APIRouter()


class RefineRequest(BaseModel):
    session_id: str
    user_prompt: str = Field(..., description="The student's question or modification request")
    current_cells: Dict[str, str] = Field(..., description="The current state of the 12 DEAP cells")


class RefineResponse(BaseModel):
    status: str
    cells: Dict[str, str]
    cells_modified: List[str]
    tutor_explanation: str
    version_number: int
    version_id: str


@router.post("/refine", response_model=RefineResponse)
async def refine_notebook(request: RefineRequest, db: Session = Depends(get_db)):
    """
    Refine an existing notebook. Creates a new immutable version.
    Previous version is never overwritten.
    """
    logger.info(f"[/refine] session={request.session_id} prompt='{request.user_prompt[:60]}'")
    try:
        svc = VersionService(db)
        result = await svc.refine(
            session_id=request.session_id,
            user_prompt=request.user_prompt,
            current_cells=request.current_cells,
        )
        return RefineResponse(**result)
    except Exception as e:
        logger.exception(f"[/refine] Failed: {e}")
        try:
            svc = VersionService(db)
            active_cells_obj = svc.get_active_cells(request.session_id)
            active_cells = active_cells_obj.to_dict() if active_cells_obj else {}
            notebook = svc.notebook_repo.get_notebook_by_session(request.session_id)
            active_ver_num = 1
            active_ver_id = ""
            if notebook and notebook.active_version_id:
                active_ver = svc.version_repo.get_version_by_id(notebook.active_version_id)
                if active_ver:
                    active_ver_num = active_ver.version_number
                    active_ver_id = active_ver.version_id

            return RefineResponse(
                status="reverted",
                cells=active_cells,
                cells_modified=[],
                tutor_explanation=f"Refinement pipeline error: {str(e)}. Automatically rolled back to the previous working version.",
                version_number=active_ver_num,
                version_id=active_ver_id,
            )
        except Exception as fallback_err:
            logger.exception(f"[/refine] Rollback fallback failed: {fallback_err}")
            raise HTTPException(status_code=500, detail=str(e))
