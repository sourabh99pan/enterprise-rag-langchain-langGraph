
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class ReviewState(TypedDict):
    draft_answer: str
    final_answer: str


def generate_draft(state: ReviewState):
    # Simulate an AI-generated answer
    draft = state["draft_answer"]

    print(f"AI-generated draft: {draft}")

    return {}


def human_review(state: ReviewState):
    # Pause and ask the human to review or edit the draft
    reviewed_answer = interrupt(
        {
            "message": "Review the AI-generated answer.",
            "draft_answer": state["draft_answer"],
            "instruction": (
                "Return the corrected answer as a string."
            ),
        }
    )

    return {"final_answer": reviewed_answer}


def show_final_answer(state: ReviewState):
    print(f"\nFinal answer: {state['final_answer']}")
    return {}


builder = StateGraph(ReviewState)

builder.add_node("generate_draft", generate_draft)
builder.add_node("human_review", human_review)
builder.add_node("show_final_answer", show_final_answer)

builder.add_edge(START, "generate_draft")
builder.add_edge("generate_draft", "human_review")
builder.add_edge("human_review", "show_final_answer")
builder.add_edge("show_final_answer", END)

graph = builder.compile(checkpointer=InMemorySaver())

config = {
    "configurable": {
        "thread_id": "human-edit-demo-001"
    }
}

result = graph.invoke(
    {
        "draft_answer": (
            "Employees can carry forward 10 annual leave days."
        ),
        "final_answer": "",
    },
    config=config,
)

print("\nGraph paused for human review.")
print("Interrupt information:", result["__interrupt__"])

corrected_answer = input(
    "\nEnter the corrected answer: "
).strip()

result = graph.invoke(
    Command(resume=corrected_answer),
    config=config,
)

print("\nWorkflow completed.")
print("Final state:", result)
