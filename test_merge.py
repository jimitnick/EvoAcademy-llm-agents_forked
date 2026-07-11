import asyncio
import operator
from typing import TypedDict, Annotated, Dict
from langgraph.graph import StateGraph, START, END

class MyState(TypedDict):
    notebook_cells: Annotated[Dict[str, str], operator.ior]

def test_node(state: MyState):
    return {"notebook_cells": {"b": "new_b", "c": "new_c"}}

graph = StateGraph(MyState)
graph.add_node("test", test_node)
graph.add_edge(START, "test")
graph.add_edge("test", END)
compiled = graph.compile()

async def main():
    state = {"notebook_cells": {"a": "old_a", "b": "old_b"}}
    res = await compiled.ainvoke(state)
    print(res)

asyncio.run(main())
