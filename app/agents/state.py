from typing import TypedDict , Annotated,Dict,List  
import operator


class CoderState(TypedDict):
    cell_name:str
    cell_prompt:str
    user_prompt:str
    generated_code:str
    attempts:int
    is_valid:bool
    error_msg:str

    notebook_cells:Dict[str,str]

class NotebookState(TypedDict):
    user_prompt:str
    target_problem:str
    subtask_prompts:Dict[str,str]
    notebook_cells:Annotated[Dict[str,str],operator.ior]
    compiled_script:str

    ast_dependency_map: Dict[str, dict]
    cells_to_modify: List[str]
    needs_understanding: bool
    is_traceback_error: bool
    traceback_msg: str
