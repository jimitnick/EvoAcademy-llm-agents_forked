import ast
from typing import Dict, Set

# Visitor targeting definitions and method calls
class DependencyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.defined_names = set()
        self.called_names = set()

    def visit_FunctionDef(self, node):
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined_names.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.called_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.called_names.add(node.func.attr)
        self.generic_visit(node)

# Scans notebooks to build a dictionary of dependencies
def analyze_notebook_dependencies(notebook_cells: Dict[str, str]) -> Dict[str, dict]:
    dependency_map = {}

    for cell_name, code in notebook_cells.items():
        if not code or code.startswith("# ERROR"):
            continue
        try:
            tree = ast.parse(code)
            visitor = DependencyVisitor()
            visitor.visit(tree)

            dependency_map[cell_name] = {
                "defines": list(visitor.defined_names),
                "calls": list(visitor.called_names)
            }
        except SyntaxError:
            dependency_map[cell_name] = {"defines": [], "calls": [], "error": "SyntaxError"}
    return dependency_map