from pydantic import BaseModel, Field
from typing import List
from app.agents.state import NotebookState
from app.core.llm import architect_llm
from app.utils.ast_parser import analyze_notebook_dependencies

class RoutingDecision(BaseModel):
    cells_to_modify: List[str] = Field(description="Exact keys of the cells that must be rewritten (e.g., ['crossover', 'toolbox']). Empty if no code changes needed.")
    needs_understanding: bool = Field(description="True if the user is asking an educational question (e.g., 'How does this work?').")

def dependency_analyzer_node(state: NotebookState):
    print("--> [Analyzer] Evaluating user request for modifications...")

    current_cells = state.get("notebook_cells", {})
    user_request = state.get("user_prompt", "")
    
    ast_map = analyze_notebook_dependencies(current_cells)
    
    router_llm = architect_llm.with_structured_output(RoutingDecision)
    
    system_prompt = f"""
    You are the Dependency Analyzer for a DEAP Evolutionary Algorithm codebase.
    The student has requested: "{user_request}"
    
    Here is the exact structure of their current code (what each cell defines and calls):
    {ast_map}
    
    Your task:
    1. Determine if the student is asking a theoretical question (needs_understanding = True).
    2. Determine EXACTLY which cells need to be modified to fulfill a code change.
    3. CRITICAL: If a function definition changes in one cell (e.g., 'evaluation'), you MUST also flag the cell where it is registered or called (e.g., 'toolbox').
    """
    
    decision = router_llm.invoke(system_prompt)
    
    print(f"    -> [Analyzer] Flagged cells for modification: {decision.cells_to_modify}")
    if decision.needs_understanding:
        print(f"    -> [Analyzer] Flagged as an educational query.")
    
    return {
        "ast_dependency_map": ast_map,
        "cells_to_modify": decision.cells_to_modify,
        "needs_understanding": decision.needs_understanding
    }


def modifier_agent_node(state: NotebookState):
    print(f"--> [Modifier Agent] Rewriting cells: {state.get('cells_to_modify')}")
    # We will build the AST-aware code generation here later
    return state

def fixer_agent_node(state: NotebookState):
    print(f"--> [Fixer Agent] Analyzing Traceback: {state.get('traceback_msg')}")
    # We will build the error-correction logic here later
    return state

def learner_agent_node(state: NotebookState):
    print("--> [Learner Agent] Generating educational explanation...")
    # We will build the RAG/Explanation logic here later
    return state