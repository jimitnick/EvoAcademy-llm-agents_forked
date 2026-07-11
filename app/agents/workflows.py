from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from app.agents.state import NotebookState, CoderState, DebugState
from app.agents.nodes.generation import task_splitter_node, parallel_coder_node, prompt_guardrail_node
from app.agents.nodes.verification import unit_verifier_node

DEAP_CELLS = [
    "imports", "config", "creator", "evaluation", "crossover", "mutation", "selection",
    "initialization", "toolbox", "main_algorithm", "stats", "visualization"
]



# Route coder based on syntax validity
def route_verification(state: CoderState):
    if state.get("is_valid"):
        return END
    elif state.get("attempts", 0) >= 3:
        print(f"Max attempts reached for {state['cell_name']}.")
        return END
    else:
        return "coder"

# Coder sub-graph declaration
coder_workflow = StateGraph(CoderState)
coder_workflow.add_node("coder", parallel_coder_node)
coder_workflow.add_node("verifier", unit_verifier_node)

coder_workflow.add_edge(START, "coder")
coder_workflow.add_edge("coder", "verifier")
coder_workflow.add_conditional_edges("verifier", route_verification, ["coder", END])

coder_subgraph = coder_workflow.compile()

# Send prompts to parallel subgraphs
def orchestrate_parallel_coders(state: NotebookState):
    return [
        Send("coder_subgraph", {
            "cell_name": name,
            "cell_prompt": prompt,
            "user_prompt": state["user_prompt"],
            "attempts": 0,
            "is_valid": False
        })
        for name, prompt in state["subtask_prompts"].items()
    ]

# Concatenates output cells into single script
def orchestrator_node(state: NotebookState):
    print("--> [Orchestrator] Assembling final DEAP script...")
    generated_cells = state.get("notebook_cells", {})
    assembled_code_blocks = []
    for cell_name in DEAP_CELLS:
        code = generated_cells.get(cell_name)
        if code:
            header = f"\n# {'='*40}\n# CELL: {cell_name.upper()}\n# {'='*40}\n"
            assembled_code_blocks.append(header + code)
        else:
            print(f"    -> [WARNING] Missing cell: {cell_name}.")
            assembled_code_blocks.append(f"\n# ERROR: Agent failed to generate {cell_name} cell.\n")
    final_script = "\n".join(assembled_code_blocks)
    return {"compiled_script": final_script}

# Main generation graph workflow
workflow = StateGraph(NotebookState)
workflow.add_node("task_splitter", task_splitter_node)
workflow.add_node("coder_subgraph", coder_subgraph)
workflow.add_node("orchestrator", orchestrator_node)

workflow.add_edge(START, "task_splitter")
workflow.add_conditional_edges("task_splitter", orchestrate_parallel_coders, ["coder_subgraph"])
workflow.add_edge("coder_subgraph", "orchestrator")
workflow.add_edge("orchestrator", END)

generate_graph = workflow.compile()

from app.agents.nodes.refinement import (
    dependency_analyzer_node,
    modifier_agent_node,
    fixer_agent_node,
    learner_agent_node
)

# Routes call to debugger, modifier, or tutor agents
def route_post_analysis(state: NotebookState) -> str:
    if state.get("is_traceback_error"):
        return "fixer_agent"
    elif len(state.get("cells_to_modify", [])) > 0:
        return "modifier_agent"
    elif state.get("needs_understanding"):
        return "learner_agent"
    return END

# Refinement graph workflow
refine_workflow = StateGraph(NotebookState)
refine_workflow.add_node("dependency_analyzer", dependency_analyzer_node)
refine_workflow.add_node("modifier_agent", modifier_agent_node)
refine_workflow.add_node("fixer_agent", fixer_agent_node)
refine_workflow.add_node("learner_agent", learner_agent_node)

refine_workflow.add_edge(START, "dependency_analyzer")
refine_workflow.add_conditional_edges(
    "dependency_analyzer", 
    route_post_analysis, 
    {
        "fixer_agent": "fixer_agent",
        "modifier_agent": "modifier_agent",
        "learner_agent": "learner_agent",
        END: END
    }
)

refine_workflow.add_edge("modifier_agent", END)
refine_workflow.add_edge("fixer_agent", END)
refine_workflow.add_edge("learner_agent", END)

refine_graph = refine_workflow.compile()

def route_gatekeeper(state:NotebookState)->str:
    if state.get("is_valid_ea_prompt"):
        return "task_splitter"
    return END

main_workflow = StateGraph(NotebookState)

main_workflow.add_node("prompt_guardrail",prompt_guardrail_node)
main_workflow.add_node("task_splitter",task_splitter_node)
main_workflow.add_node("coder_subgraph",coder_subgraph)
main_workflow.add_node("orchestrator",orchestrator_node)

main_workflow.add_edge(START,"prompt_guardrail")

main_workflow.add_conditional_edges("prompt_guardrail",route_gatekeeper,{
    "task_splitter":"task_splitter",
    END:END
})

main_workflow.add_conditional_edges("task_splitter",orchestrate_parallel_coders,["coder_subgraph"])
main_workflow.add_edge("coder_subgraph","orchestrator")
main_workflow.add_edge("orchestrator",END)

generate_graph = main_workflow.compile()

# --- DEBUG WORKFLOW ---
def route_debug_start(state: DebugState) -> str:
    if state.get("traceback_msg"):
        return "fixer_agent"
    return END

def prepare_for_verification(state: DebugState):
    # Concatenate updated cells to verify syntax
    compiled = "\n".join([code for code in state.get("notebook_cells", {}).values() if code and not code.startswith("# ERROR")])
    return {"generated_code": compiled, "cell_name": "debug_script", "attempts": state.get("attempts", 0) + 1}

def route_debug_verification(state: DebugState) -> str:
    if state.get("is_valid"):
        return END
    elif state.get("attempts", 0) >= 3:
        print("Max debug attempts reached.")
        return END
    else:
        return "prepare_fixer_retry"

def prepare_fixer_retry(state: DebugState):
    return {"traceback_msg": state.get("error_msg", "")}

debug_workflow = StateGraph(DebugState)
debug_workflow.add_node("fixer_agent", fixer_agent_node)
debug_workflow.add_node("prepare_for_verification", prepare_for_verification)
debug_workflow.add_node("unit_verifier", unit_verifier_node)
debug_workflow.add_node("prepare_fixer_retry", prepare_fixer_retry)

debug_workflow.add_conditional_edges(START, route_debug_start, ["fixer_agent", END])
debug_workflow.add_edge("fixer_agent", "prepare_for_verification")
debug_workflow.add_edge("prepare_for_verification", "unit_verifier")
debug_workflow.add_conditional_edges("unit_verifier", route_debug_verification, {
    END: END,
    "prepare_fixer_retry": "prepare_fixer_retry"
})
debug_workflow.add_edge("prepare_fixer_retry", "fixer_agent")

debug_graph = debug_workflow.compile()