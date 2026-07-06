from langgraph.graph import StateGraph,START,END
from langgraph.constants import Send
from app.agents.state import NotebookState,CoderState
from app.agents.nodes.generation import task_splitter_node,parallel_coder_node
from app.agents.nodes.verification import unit_verifier_node


DEAP_CELLS = [
    "imports", "config", "creator", "evaluation", "crossover", "mutation", "selection",
    "initialization", "toolbox", "main_algorithm", "stats", "visualization"
]

def route_verification(state:CoderState):
    if state.get("is_valid"):
        return END
    elif state.get("attempts",0)>=3:
        print(f"Max attempts reached for {state["cell_name"]}.")
        return END
    else:
        return "coder"
    

coder_workflow = StateGraph(CoderState)
coder_workflow.add_node("coder",parallel_coder_node)
coder_workflow.add_node("verifier",unit_verifier_node)

coder_workflow.add_edge(START,"coder")
coder_workflow.add_edge("coder","verifier")
coder_workflow.add_conditional_edges("verifier",route_verification,["coder",END])

coder_subgraph = coder_workflow.compile()



def orchestrate_parallel_coders(state:NotebookState):
    return [
        Send("coder_subgraph",{
            "cell_name":name,
            "cell_prompt":prompt,
            "user_prompt":state["user_prompt"],
            "attempts":0,
            "is_valid":False
        })
        for name,prompt in state["subtask_prompts"].items()
    ]


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

workflow = StateGraph(NotebookState)

workflow.add_node("task_splitter",task_splitter_node)
workflow.add_node("coder_subgraph",coder_subgraph)
workflow.add_node("orchestrator",orchestrator_node)

workflow.add_edge(START,"task_splitter")
workflow.add_conditional_edges("task_splitter",orchestrate_parallel_coders,["coder_subgraph"])
workflow.add_edge("coder_subgraph","orchestrator")
workflow.add_edge("orchestrator",END)

generate_graph = workflow.compile()


from app.agents.nodes.refinement import (
    dependency_analyzer_node,
    modifier_agent_node,
    fixer_agent_node,
    learner_agent_node
)


def route_post_analysis(state: NotebookState) -> str:
    """
    The decision diamonds from the architecture diagram.
    Routes the state to the correct specialized agent.
    """
    if state.get("is_traceback_error"):
        return "fixer_agent"
    
    elif len(state.get("cells_to_modify", [])) > 0:
        return "modifier_agent"
        
    elif state.get("needs_understanding"):
        return "learner_agent"
        
    return END

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
