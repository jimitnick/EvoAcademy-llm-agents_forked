from typing import TypedDict, Annotated, Dict, List  
import operator

# State for parallel Junior Coder subgraph
class CoderState(TypedDict):
    cell_name: str
    cell_prompt: str
    user_prompt: str
    generated_code: str
    attempts: int
    is_valid: bool
    error_msg: str
    notebook_cells: Dict[str, str]

# Custom state merger to avoid duplicate write conflicts
def reduce_user_prompt(old: str, new: str) -> str:
    return new if new else old

# Main state representation for the notebook flow
class NotebookState(TypedDict):
    session_id: str
    user_prompt: Annotated[str, reduce_user_prompt]
    target_problem: str
    subtask_prompts: Dict[str, str]
    notebook_cells: Annotated[Dict[str, str], operator.ior]
    compiled_script: str
    ast_dependency_map: Dict[str, dict]
    cells_to_modify: List[str]
    needs_understanding: bool
    is_traceback_error: bool
    traceback_msg: str
    educational_response: str
