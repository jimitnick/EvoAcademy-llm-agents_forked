import ast
from app.agents.state import CoderState

# Node performing static syntax verification using ast parsing
def unit_verifier_node(state: CoderState):
    code = state.get("generated_code", "")
    cell_name = state.get("cell_name", "unknown")
    
    print(f"[VERIFIER] Checking syntax for {cell_name}...")

    try:
        # Check syntax errors by building the AST tree
        ast.parse(code)
        return {
            "is_valid": True,
            "error_msg": "",
            "notebook_cells": {cell_name: code}
        }
    except SyntaxError as e:
        print(f"[VERIFIER] Syntax error in {cell_name}: {str(e)}")
        return {
            "is_valid": False,
            "error_msg": f"SyntaxError: {str(e)}\nFix the code and return only valid Python."
        }
