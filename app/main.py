from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging

from app.agents.workflows import generate_graph, refine_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI entry gateway app configuration
app = FastAPI(title="LLM API Gateway")

# Authorized endpoints requests origins setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    session_id: str = Field(..., description="Unique ID for the learning session")
    prompt: str = Field(..., description="The student's raw prompt for the EA problem")

class GenerateResponse(BaseModel):
    status: str
    target_problem: str
    cells: Dict[str, str]
    compiled_script: str

class RefineRequest(BaseModel):
    session_id: str
    user_prompt: str = Field(..., description="The student's question or modification request")
    current_cells: Dict[str, str] = Field(..., description="The current state of the 12 DEAP cells")

class RefineResponse(BaseModel):
    status: str
    cells: Dict[str, str]
    cells_modified: List[str]
    tutor_explanation: str

class DebugRequest(BaseModel):
    session_id: str
    traceback_msg: str = Field(..., description="The runtime error thrown by the Jupyter kernel")
    current_cells: Dict[str, str] = Field(..., description="The current state of the 12 DEAP cells")

# Check gateway server connectivity
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llm-agents"}

# Splitting tasks & concurrency coder pipeline
@app.post("/generate", response_model=GenerateResponse)
async def generate_notebook(request: GenerateRequest):
    logger.info(f"Received GENERATE request for session: {request.session_id}")
    initial_state = {
        "session_id": request.session_id,
        "user_prompt": request.prompt,
        "target_problem": "",
        "subtask_prompts": {},
        "notebook_cells": {},
        "compiled_script": ""
    }
    try:
        result = await generate_graph.ainvoke(initial_state)
        return GenerateResponse(
            status="success",
            target_problem=result.get("target_problem", "Unknown"),
            cells=result.get("notebook_cells", {}),
            compiled_script=result.get("compiled_script", "")
        )
    except Exception as e:
        logger.error(f"Generation pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Router node executing modifications or CS explanations
@app.post("/refine", response_model=RefineResponse)
async def refine_notebook(request: RefineRequest):
    logger.info(f"Received REFINE request for session: {request.session_id}") 
    initial_state = {
        "session_id": request.session_id,
        "user_prompt": request.user_prompt,
        "notebook_cells": request.current_cells,
        "compiled_script": "",
        "ast_dependency_map": {},
        "cells_to_modify": [],
        "needs_understanding": False,
        "is_traceback_error": False,  
        "traceback_msg": "",
        "educational_response": ""
    }
    try:
        result = await refine_graph.ainvoke(initial_state)
        return RefineResponse(
            status="success",
            cells=result.get("notebook_cells", {}),
            cells_modified=result.get("cells_to_modify", []),
            tutor_explanation=result.get("educational_response", "")
        )
    except Exception as e:
        logger.error(f"Refinement pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Fixer debugger endpoint patching exceptions tracebacks
@app.post("/debug", response_model=RefineResponse)
async def debug_notebook(request: DebugRequest):
    logger.info(f"Received DEBUG request for session: {request.session_id}")
    initial_state = {
        "session_id": request.session_id,
        "user_prompt": "Fix the following runtime error.", 
        "notebook_cells": request.current_cells,
        "compiled_script": "",
        "ast_dependency_map": {},
        "cells_to_modify": [],
        "needs_understanding": False,
        "is_traceback_error": True, 
        "traceback_msg": request.traceback_msg,
        "educational_response": ""
    }
    try:
        result = await refine_graph.ainvoke(initial_state)
        return RefineResponse(
            status="success",
            cells=result.get("notebook_cells", {}),
            cells_modified=[],  # modified cell updates are saved directly in-place
            tutor_explanation="Code has been patched based on the runtime traceback."
        )
    except Exception as e:
        logger.error(f"Debug pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
