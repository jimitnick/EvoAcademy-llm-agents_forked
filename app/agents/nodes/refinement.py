from pydantic import BaseModel, Field
from typing import List, Dict
from app.agents.state import NotebookState
from app.core.llm import architect_llm
from app.utils.ast_parser import analyze_notebook_dependencies
from app.core.memory import mem0_client

# structured schema for modified cell rewrites
class ModifiedCells(BaseModel):
    updated_cells: Dict[str, str] = Field(description="A dictionary where the key is the cell name and the value is the newly rewritten Python code.")

# structured schema to route user request type
class RoutingDecision(BaseModel):
    cells_to_modify: List[str] = Field(description="Exact keys of the cells that must be rewritten (e.g., ['crossover', 'toolbox']). Empty if no code changes needed.")
    needs_understanding: bool = Field(description="True if the user is asking an educational question (e.g., 'How does this work?').")

# Analyzer categorizes user prompts and evaluates dependencies via AST
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

# Modifier agent executes surgical code rewrites
def modifier_agent_node(state: NotebookState):
    cells_to_edit = state.get("cells_to_modify", [])
    print(f"--> [Modifier Agent] Initiating surgical rewrite for: {cells_to_edit}")
    
    if not cells_to_edit:
        return state
    
    current_code_context = {cell: state["notebook_cells"].get(cell, "") for cell in cells_to_edit}
    modifier_llm = architect_llm.with_structured_output(ModifiedCells)
    system_prompt = f"""
    You are surgical code modifier for a DEAP Evolutionary Algorithm codebase. The student has requested this change:"{state['user_prompt']}"
    You have been authorized to modify only the following cells :{cells_to_edit}
    Current code for these cells:{current_code_context}
    Global Notebook AST Dependency Map:{state['ast_dependency_map']}
    INSTRUCTIONS:
    1. Rewrite the code for the authorized cells to fulfill the user's request.
    2. Use the AST map to ensure your new  variable/function names do not break references in other cells.
    3. Return only valid Python code for each cell. Do not include markdown blocks.
    4. Format your output as a dictionary mapping cell keys to python code.
    """
    result = modifier_llm.invoke(system_prompt)
    print(f" --> [Modifier Agent] Successfully rewritten {list(result.updated_cells.keys())}")
    return {"notebook_cells": result.updated_cells}

# Fixer debugger agent handles runtime tracebacks
def fixer_agent_node(state: NotebookState):
    print(f"--> [Fixer Agent] Diagnosing runtime traceback ...")
    traceback = state.get("traceback_msg", "")
    if not traceback:
        return state
    
    fixer_llm = architect_llm.with_structured_output(ModifiedCells)
    system_prompt = f"""
    You are an expert Python debugger specializing in the DEAP Evolutionary Algorithm library.
    The student's compiled notebook threw the following runtime error in their Jupyter Kernel:
    ===TRACEBACK===
    {traceback}
    ===============
    Here is their current codebase:{state['notebook_cells']}

    INSTRUCTIONS:
    1.Analyze the traceback to pinpoint the exact cell(s) causing the failure.
    2.Write the corrected code for only those specific cells.
    3.Do not rewrite cells that are functioning correctly.
    4.Return only valid Python code for the fixed cells.
    """
    result = fixer_llm.invoke(system_prompt)
    print(f"--> [Fixer Agent] Successfully patched errors in :{list(result.updated_cells.keys())}")
    return {"notebook_cells": result.updated_cells}

# CS Tutor agent retrieves and saves memory profiles from Mem0
def learner_agent_node(state: NotebookState):
    print("--> [Learner Agent] Generating educational explanation...")
    user_query = state.get("user_prompt", "")
    current_code = state.get("notebook_cells", {})
    session_id = state.get("session_id", "default_session")
    
    # Retrieve past student profiles from memory
    memories = []
    try:
        memories = mem0_client.get_all(user_id=session_id)
    except Exception as e:
        print(f"Failed to fetch memories from Mem0: {e}")
        
    memory_context = "\n".join([f"- {m['text']}" for m in memories]) if memories else "No past memory of this student."

    system_prompt = f"""
    You are a Computer Science Tutor specializing in Evolutionary Algorithms and DEAP.
    
    Student Past Interactions/Profile:
    {memory_context}
    
    The student asked the following question: "{user_query}"
    Here is the current state of their notebook:{current_code}
    
    INSTRUCTIONS:
    1. Answer their question clearly, concisely, and educationally.
    2. Reference specific variable names or mechanics in their current code if it helps illustrate the concept (e.g., point out their current mutation rate).
    3. Do Not write full script rewrites. You are a tutor, not a code generator. Use small inline code snippets only if necessary for explanation.
    4. Format your response in clean Markdown.
    """
    response = architect_llm.invoke(system_prompt)
    
    # Store questions and explanations in long-term context vector database
    try:
        mem0_client.add(f"Student asked: {user_query}", user_id=session_id)
        mem0_client.add(f"Tutor summarized: {response.content[:150]}", user_id=session_id)
    except Exception as e:
        print(f"Failed to add memory to Mem0: {e}")

    print(f"--> [Learner Agent] Explanation generated.")
    return {"educational_response": response.content}