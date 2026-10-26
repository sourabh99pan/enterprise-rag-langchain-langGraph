
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ---------- Shared state ----------

class WorkflowState(TypedDict):
    question: str
    context: str
    answer: str


# ---------- Child graph nodes ----------

def retrieve(state: WorkflowState):
    print("Child: Retrieving policy information")

    # Simulate document retrieval
    return {
        "context": (
            "Employees receive 20 annual leave days per year."
        )
    }


def generate(state: WorkflowState):
    print("Child: Generating answer")

    return {
        "answer": (
            f"Question: {state['question']}\n"
            f"Answer: {state['context']}"
        )
    }


# ---------- Build child graph ----------

child_builder = StateGraph(WorkflowState)

child_builder.add_node("retrieve", retrieve)
child_builder.add_node("generate", generate)

child_builder.add_edge(START, "retrieve")
child_builder.add_edge("retrieve", "generate")
child_builder.add_edge("generate", END)

child_graph = child_builder.compile()


# ---------- Build parent graph ----------

parent_builder = StateGraph(WorkflowState)

# Register the compiled child graph directly
parent_builder.add_node("rag_subgraph", child_graph)

parent_builder.add_edge(START, "rag_subgraph")
parent_builder.add_edge("rag_subgraph", END)

parent_graph = parent_builder.compile()


# ---------- Execute parent graph ----------

result = parent_graph.invoke({
    "question": "How many annual leave days do employees receive?",
    "context": "",
    "answer": ""
})

print("\nFinal answer:")
print(result["answer"])
