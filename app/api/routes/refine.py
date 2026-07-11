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
        raise HTTPException(status_code=500, detail=str(e))
