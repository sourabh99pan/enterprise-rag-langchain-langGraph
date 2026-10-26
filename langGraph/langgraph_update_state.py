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

# Step 1: Run the graph
result = graph.invoke({"count": 0}, config=config)
print("Initial result:", result)

# Step 2: Inspect saved state
print("Before update:", graph.get_state(config).values)

# Step 3: Update the saved state
graph.update_state(config, {"count": 10})

# Step 4: Inspect the updated state
print("After update:", graph.get_state(config).values)

result = graph.invoke(None, config=config)

print("After resuming:", result)