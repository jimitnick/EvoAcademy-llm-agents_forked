import ast
from app.agents.state import CoderState

def unit_verifier_node(state:CoderState):
    code = state.get("generated_code","")
    cell_name=state.get("cell_name","unknown")
    
    print(f"[VERIFIER] Checking syntax for {cell_name}...")

    try:
        ast.parse(code)
        return {
            "is_valid":True,
            "error_msg":"",
            "notebook_cells":{cell_name:code}
        }
    
    except SyntaxError:
        print(f"[VERIFIER] Syntax error in {cell_name}: {str(SyntaxError)}")
        return{
            "is_valid":False,
            "error_msg":f"SyntaxError: {str(SyntaxError)}\nFix the code and return only valid Python."
        }
