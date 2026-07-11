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
        raise HTTPException(status_code=500, detail=str(e))
