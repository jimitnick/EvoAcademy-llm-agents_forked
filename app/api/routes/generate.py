"""POST /generate — Generate a brand-new EA notebook."""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.version_service import VersionService

logger = logging.getLogger(__name__)
router = APIRouter()


class GenerateRequest(BaseModel):
    session_id: str = Field(..., description="Unique ID for the learning session")
    prompt: str = Field(..., description="The student's raw prompt for the EA problem")


class GenerateResponse(BaseModel):
    status: str
    target_problem: str
    cells: Dict[str, str]
    compiled_script: str
    version_number: int
    version_id: str


@router.post("/generate", response_model=GenerateResponse)
async def generate_notebook(request: GenerateRequest, db: Session = Depends(get_db)):
    """
    Generate a brand-new evolutionary algorithm notebook.
    Creates version_1.ipynb in storage/notebooks/session_{id}/.
    Any previous history for this session is cleared.
    """
    logger.info(f"[/generate] session={request.session_id} prompt='{request.prompt[:60]}'")
    try:
        svc = VersionService(db)
        result = await svc.generate(session_id=request.session_id, prompt=request.prompt)
        return GenerateResponse(**result)
    except ValueError as e:
        logger.warning(f"[/generate] Validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"[/generate] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
