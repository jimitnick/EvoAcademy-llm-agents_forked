from pydantic import BaseModel, Field
from app.agents.state import NotebookState, CoderState
from app.core.llm import architect_llm, coder_llm

# Structured schema mapping the 12 DEAP cells
class Subtasks(BaseModel):
    target_problem: str = Field(description="The formal name of the problem")
    imports: str = Field(description="Prompt for the imports cell")
    config: str = Field(description="Prompt for the config cell")
    creator: str = Field(description="Prompt for the creator cell")
    evaluation: str = Field(description="Prompt for the evaluation cell")
    crossover: str = Field(description="Prompt for the crossover cell")
    mutation: str = Field(description="Prompt for the mutation cell")
    selection: str = Field(description="Prompt for the selection cell")
    initialization: str = Field(description="Prompt for the initialization cell")
    toolbox: str = Field(description="Prompt for the toolbox cell")
    main_algorithm: str = Field(description="Prompt for the main_algorithm cell")
    stats: str = Field(description="Prompt for the stats cell")
    visualization: str = Field(description="Prompt for the visualization cell")

# Splits query into 12 detailed prompts
def task_splitter_node(state: NotebookState):
    structured_llm = architect_llm.with_structured_output(Subtasks)

    system_prompt = f"""You are an expert in Evolutionary algorithms and the DEAP Python library.
    The user wants to solve: {state['user_prompt']}
    Break this down into 12 distinct, highly specific prompts. Each prompt will be sent to a separate junior coder agent.
    Ensure the prompts instruct the coders to use compatible variables and data structures.
    """
    result = structured_llm.invoke(system_prompt)
    subtasks = result.model_dump()
    problem = subtasks.pop("target_problem")
    return {"target_problem": problem, "subtask_prompts": subtasks}

# Async node writing code blocks concurrently
async def parallel_coder_node(state: CoderState):
    attempts = state.get("attempts", 0) + 1
    error_context = f"\nPREVIOUS ERROR TO FIX:{state['error_msg']}" if state.get("error_msg") else ""

    system_prompt = f"""You are a specialized Python developer writing code for a DEAP Evolutionary Algorithm.
    You are writing only the `{state['cell_name']}` cell for a {state['user_prompt']}.
    
    INSTRUCTION:{state['cell_prompt']}{error_context}

    Return only valid Python code. Do not use markdown blocks. Do not explain the code.
    """
    response = await coder_llm.ainvoke(system_prompt)
    return {
        "generated_code": response.content.strip(),
        "attempts": attempts
    }
