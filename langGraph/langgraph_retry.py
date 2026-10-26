
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict):
    question: str
    attempts: int
    context: str
    answer: str


def retrieve_documents(state: GraphState):
    attempts = state["attempts"] + 1

    # Simulate better results after retrying
    if attempts == 1:
        context = "No useful policy information found."
    else:
        context = "Domestic hotel reimbursement is up to ₹4,000 per night."

    print(f"Retrieval attempt: {attempts}")

    return {
        "attempts": attempts,
        "context": context
    }


def check_retrieval(state: GraphState):
    if "No useful" not in state["context"]:
        return "generate"

    if state["attempts"] < 2:
        return "retry"

    return "fallback"


def generate_answer(state: GraphState):
    return {
        "answer": f"Based on the policy: {state['context']}"
    }


def fallback_answer(state: GraphState):
    return {
        "answer": "I don't know based on the available policy documents."
    }


builder = StateGraph(GraphState)

builder.add_node("retrieve", retrieve_documents)
builder.add_node("generate", generate_answer)
builder.add_node("fallback", fallback_answer)

builder.add_edge(START, "retrieve")

builder.add_conditional_edges(
    "retrieve",
    check_retrieval,
    {
        "generate": "generate",
        "retry": "retrieve",
        "fallback": "fallback"
    }
)

builder.add_edge("generate", END)
builder.add_edge("fallback", END)

graph = builder.compile()

result = graph.invoke({
    "question": "What is the hotel reimbursement limit?",
    "attempts": 0,
    "context": "",
    "answer": ""
})

print("\nFinal result:")
print(result)
