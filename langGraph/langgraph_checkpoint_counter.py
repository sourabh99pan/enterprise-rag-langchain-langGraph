from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class GraphState(TypedDict):
    count: int


def increment_counter(state: GraphState):
    return {"count": state["count"] + 1}


builder = StateGraph(GraphState)

builder.add_node("increment", increment_counter)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

config = {
    "configurable": {
        "thread_id": "counter-001"
    }
}

result = graph.invoke({"count": 1}, config=config)
print("First result:", result)

snapshot = graph.get_state(config)
print("Saved state:", snapshot.values)

result = graph.invoke(
    {"count": result["count"]},
    config=config
)

print("Second result:", result)

snapshot = graph.get_state(config)
print("Updated saved state:", snapshot.values)