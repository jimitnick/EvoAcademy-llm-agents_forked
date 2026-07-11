from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging
import uvicorn

from app.agents.workflows import generate_graph, refine_graph, DEAP_CELLS
from app.core import version_store
import sqlite3
import ast

def validate_cells(cells: Dict[str, str]) -> Optional[str]:
    for cell_name, code in cells.items():
        if code and not code.startswith("# ERROR"):
            try:
                ast.parse(code)
            except SyntaxError as e:
                return f"SyntaxError in cell '{cell_name}': {str(e)}"
    return None

def compile_cells(cells: Dict[str, str]) -> str:
    assembled_code_blocks = []
    for cell_name in DEAP_CELLS:
        code = cells.get(cell_name)
        if code:
            header = f"\n# {'='*40}\n# CELL: {cell_name.upper()}\n# {'='*40}\n"
            assembled_code_blocks.append(header + code)
        else:
            assembled_code_blocks.append(f"\n# ERROR: Agent failed to generate {cell_name} cell.\n")
    return "\n".join(assembled_code_blocks)

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

class RollbackRequest(BaseModel):
    version_number: int

# Check gateway server connectivity
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llm-agents"}

# Splitting tasks & concurrency coder pipeline
@app.post("/generate", response_model=GenerateResponse)
async def generate_notebook(request: GenerateRequest):
    logger.info(f"Received GENERATE request for session: {request.session_id}")
    
    # Start a fresh session state by deleting previous versions for this session_id
    try:
        with version_store.get_db_connection() as conn:
            conn.execute("DELETE FROM notebook_versions WHERE session_id = ?", (request.session_id,))
            conn.commit()
    except Exception as e:
        logger.warning(f"Failed to clear existing session versions: {e}")

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
        cells = result.get("notebook_cells", {})
        compiled_script = result.get("compiled_script", "")
        
        # Save Version 1 of the notebook in the database
        version_store.save_version(
            session_id=request.session_id,
            user_intent=f"Initial generation: {request.prompt}",
            cells=cells,
            compiled_script=compiled_script,
            status="working"
        )
        
        return GenerateResponse(
            status="success",
            target_problem=result.get("target_problem", "Unknown"),
            cells=cells,
            compiled_script=compiled_script
        )
    except Exception as e:
        logger.error(f"Generation pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Router node executing modifications or CS explanations
@app.post("/refine", response_model=RefineResponse)
async def refine_notebook(request: RefineRequest):
    logger.info(f"Received REFINE request for session: {request.session_id}") 
    
    # 1. Ensure a baseline working version exists
    latest_working = version_store.get_latest_working_version(request.session_id)
    if not latest_working:
        try:
            latest_working = version_store.save_version(
                session_id=request.session_id,
                user_intent="Baseline current state",
                cells=request.current_cells,
                compiled_script=compile_cells(request.current_cells),
                status="working"
            )
        except Exception as db_err:
            logger.warning(f"Failed to create baseline version: {db_err}")

    # Use the client's current_cells, falling back to db baseline if empty
    baseline_cells = request.current_cells if request.current_cells else (latest_working["cells"] if latest_working else {})

    initial_state = {
        "session_id": request.session_id,
        "user_prompt": request.user_prompt,
        "notebook_cells": baseline_cells,
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
        updated_cells = result.get("notebook_cells", {})
        cells_modified = result.get("cells_to_modify", [])
        
        # If it was an explanation request or no changes were requested, return early
        if not cells_modified and result.get("needs_understanding", False):
            return RefineResponse(
                status="success",
                cells=updated_cells,
                cells_modified=[],
                tutor_explanation=result.get("educational_response", "")
            )

        # 2. Validate the modified/updated cells
        validation_error = validate_cells(updated_cells)
        if validation_error:
            logger.warning(f"Refined code validation failed: {validation_error}")
            version_store.save_version(
                session_id=request.session_id,
                user_intent=request.user_prompt,
                cells=updated_cells,
                compiled_script=compile_cells(updated_cells),
                status="failed",
                error_message=validation_error
            )
            # Revert to last working version
            reverted_cells = latest_working["cells"] if latest_working else baseline_cells
            reverted_script = latest_working["compiled_script"] if latest_working else compile_cells(baseline_cells)
            
            # Record rollback event in db
            version_store.save_version(
                session_id=request.session_id,
                user_intent=f"Auto-rollback from error: {validation_error}",
                cells=reverted_cells,
                compiled_script=reverted_script,
                status="working"
            )
            
            return RefineResponse(
                status="reverted",
                cells=reverted_cells,
                cells_modified=[],
                tutor_explanation=f"Validation Error: {validation_error}. Automatically rolled back to the previous working version."
            )

        # 3. Validation passed: save the new working version
        compiled_script = compile_cells(updated_cells)
        version_store.save_version(
            session_id=request.session_id,
            user_intent=request.user_prompt,
            cells=updated_cells,
            compiled_script=compiled_script,
            status="working"
        )
        
        # Log update to Mem0
        try:
            from app.core.memory import mem0_client
            mem0_client.add(
                f"User refined code with prompt '{request.user_prompt}'. Cells modified: {cells_modified}.",
                user_id=request.session_id
            )
        except Exception as mem_err:
            logger.warning(f"Failed to log modification to Mem0: {mem_err}")

        return RefineResponse(
            status="success",
            cells=updated_cells,
            cells_modified=cells_modified,
            tutor_explanation=result.get("educational_response", "Modification applied successfully.")
        )

    except Exception as e:
        logger.error(f"Refinement pipeline failed: {str(e)}")
        version_store.save_version(
            session_id=request.session_id,
            user_intent=request.user_prompt,
            cells=baseline_cells,
            compiled_script=compile_cells(baseline_cells),
            status="failed",
            error_message=str(e)
        )
        # Revert to last working cells
        reverted_cells = latest_working["cells"] if latest_working else baseline_cells
        return RefineResponse(
            status="reverted",
            cells=reverted_cells,
            cells_modified=[],
            tutor_explanation=f"Pipeline error: {str(e)}. Automatically rolled back to the previous working version."
        )

# Fixer debugger endpoint patching exceptions tracebacks
@app.post("/debug", response_model=RefineResponse)
async def debug_notebook(request: DebugRequest):
    logger.info(f"Received DEBUG request for session: {request.session_id}")
    
    # 1. Ensure a baseline working version exists
    latest_working = version_store.get_latest_working_version(request.session_id)
    if not latest_working:
        try:
            latest_working = version_store.save_version(
                session_id=request.session_id,
                user_intent="Baseline current state",
                cells=request.current_cells,
                compiled_script=compile_cells(request.current_cells),
                status="working"
            )
        except Exception as db_err:
            logger.warning(f"Failed to create baseline version: {db_err}")

    # Use the client's current_cells, falling back to db baseline if empty
    baseline_cells = request.current_cells if request.current_cells else (latest_working["cells"] if latest_working else {})

    initial_state = {
        "session_id": request.session_id,
        "user_prompt": "Fix the following runtime error.", 
        "notebook_cells": baseline_cells,
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
        updated_cells = result.get("notebook_cells", {})
        
        # 2. Validate the patched cells
        validation_error = validate_cells(updated_cells)
        if validation_error:
            logger.warning(f"Debugged code validation failed: {validation_error}")
            version_store.save_version(
                session_id=request.session_id,
                user_intent=f"Debug traceback: {request.traceback_msg[:100]}",
                cells=updated_cells,
                compiled_script=compile_cells(updated_cells),
                status="failed",
                error_message=validation_error
            )
            # Revert to last working version
            reverted_cells = latest_working["cells"] if latest_working else baseline_cells
            reverted_script = latest_working["compiled_script"] if latest_working else compile_cells(baseline_cells)
            
            # Record rollback event in db
            version_store.save_version(
                session_id=request.session_id,
                user_intent=f"Auto-rollback from debug validation error: {validation_error}",
                cells=reverted_cells,
                compiled_script=reverted_script,
                status="working"
            )
            
            return RefineResponse(
                status="reverted",
                cells=reverted_cells,
                cells_modified=[],
                tutor_explanation=f"Debug attempt generated invalid code: {validation_error}. Automatically rolled back to the previous working version."
            )

        # 3. Validation passed: save the fixed code as a new working version
        compiled_script = compile_cells(updated_cells)
        version_store.save_version(
            session_id=request.session_id,
            user_intent=f"Debug traceback: {request.traceback_msg[:100]}",
            cells=updated_cells,
            compiled_script=compiled_script,
            status="working"
        )
        
        # Log update to Mem0
        try:
            from app.core.memory import mem0_client
            mem0_client.add(
                f"Fixed runtime error with debug attempt. Error msg: {request.traceback_msg[:100]}",
                user_id=request.session_id
            )
        except Exception as mem_err:
            logger.warning(f"Failed to log debug modification to Mem0: {mem_err}")

        return RefineResponse(
            status="success",
            cells=updated_cells,
            cells_modified=[],  # modified cell updates are saved directly in-place
            tutor_explanation="Code has been patched based on the runtime traceback."
        )
        
    except Exception as e:
        logger.error(f"Debug pipeline failed: {str(e)}")
        version_store.save_version(
            session_id=request.session_id,
            user_intent=f"Debug traceback failed: {request.traceback_msg[:100]}",
            cells=baseline_cells,
            compiled_script=compile_cells(baseline_cells),
            status="failed",
            error_message=str(e)
        )
        # Revert to last working cells
        reverted_cells = latest_working["cells"] if latest_working else baseline_cells
        return RefineResponse(
            status="reverted",
            cells=reverted_cells,
            cells_modified=[],
            tutor_explanation=f"Debug pipeline error: {str(e)}. Automatically rolled back to the previous working version."
        )

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    logger.info(f"Received GET HISTORY request for session: {session_id}")
    history = version_store.get_session_history(session_id)
    return {"session_id": session_id, "history": history}

@app.post("/sessions/{session_id}/rollback", response_model=RefineResponse)
async def rollback_session(session_id: str, request: RollbackRequest):
    logger.info(f"Received MANUAL ROLLBACK request for session: {session_id} to version: {request.version_number}")
    target = version_store.get_version(session_id, request.version_number)
    if not target:
        raise HTTPException(
            status_code=404, 
            detail=f"Version {request.version_number} not found for session {session_id}"
        )
    
    # Perform the rollback
    new_ver = version_store.rollback_to_version(session_id, request.version_number)
    if not new_ver:
        raise HTTPException(status_code=500, detail="Failed to perform rollback in database")
        
    return RefineResponse(
        status="success",
        cells=new_ver["cells"],
        cells_modified=list(new_ver["cells"].keys()),
        tutor_explanation=f"Successfully rolled back to version {request.version_number}."
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
