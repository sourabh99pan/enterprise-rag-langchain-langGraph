from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class GraphState(TypedDict):
    question: str
    answer: str


def answer_question(state: GraphState):
    question = state["question"].lower()

    if "leave" in question:
        answer = "Annual leave allowance is 20 days per year."
    else:
        answer = "Please check the relevant company policy."

    return {"answer": answer}


builder = StateGraph(GraphState)

builder.add_node("answer", answer_question)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

config = {
    "configurable": {
        "thread_id": "employee-001"
    }
}

# First invocation
result1 = graph.invoke(
    {
        "question": "What is my leave allowance?",
        "answer": ""
    },
    config=config
)

print("First answer:", result1["answer"])

# Inspect the saved state
snapshot = graph.get_state(config)

print("\nSaved state:")
print(snapshot.values)

# Second invocation using the same thread
result2 = graph.invoke(
    {
        "question": "Can I carry forward unused leave?",
        "answer": ""
    },
    config=config
)

print("\nSecond answer:", result2["answer"])

print("\nUpdated saved state:")
print(graph.get_state(config).values)