
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class RetryState(TypedDict):
    question: str
    attempt: int
    answerable: bool
    answer: str


def retrieve(state: RetryState):
    """Simulate retrieval that succeeds on the second attempt."""

    attempt = state["attempt"] + 1

    print(f"\nRetrieval attempt: {attempt}")

    # Simulated behavior for learning the retry pattern.
    if attempt == 1:
        print("Retrieved context: Not sufficient")
        return {
            "attempt": attempt,
            "answerable": False,
        }

    print("Retrieved context: Sufficient")
    return {
        "attempt": attempt,
        "answerable": True,
    }


def route_after_check(state: RetryState):
    """Decide whether to generate, retry, or stop."""

    if state["answerable"]:
        return "generate"

    if state["attempt"] < 2:
        return "retrieve"

    return "fallback"


def generate_answer(state: RetryState):
    print("Generating answer...")
    return {"answer": "Answer generated from sufficient context."}


def fallback(state: RetryState):
    print("Maximum attempts reached.")
    return {"answer": "I don't know based on the available documents."}


builder = StateGraph(RetryState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate_answer)
builder.add_node("fallback", fallback)

builder.add_edge(START, "retrieve")

builder.add_conditional_edges(
    "retrieve",
    route_after_check,
    {
        "retrieve": "retrieve",
        "generate": "generate",
        "fallback": "fallback",
    },
)

builder.add_edge("generate", END)
builder.add_edge("fallback", END)

graph = builder.compile()

result = graph.invoke(
    {
        "question": "Explain the policy",
        "attempt": 0,
        "answerable": False,
        "answer": "",
    }
)

print("\nFinal result:")
print(result["answer"])
