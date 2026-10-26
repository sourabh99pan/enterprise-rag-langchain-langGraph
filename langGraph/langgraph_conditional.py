
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define the shared state
class GraphState(TypedDict):
    question: str
    documents_found: bool
    answer: str


# 2. Node: simulate document retrieval
def retrieve_documents(state: GraphState):
    question = state["question"]

    # Simplified simulation for learning
    found = "hotel" in question.lower()

    return {"documents_found": found}


# 3. Node: generate an answer
def generate_answer(state: GraphState):
    return {
        "answer": "The hotel reimbursement limit is ₹4,000 per night."
    }


# 4. Node: handle missing documents
def no_documents(state: GraphState):
    return {
        "answer": "I don't know based on the available policy documents."
    }


# 5. Routing function
def decide_next_step(state: GraphState):
    if state["documents_found"]:
        return "generate"
    return "fallback"


# 6. Build the graph
builder = StateGraph(GraphState)

builder.add_node("retrieve", retrieve_documents)
builder.add_node("generate", generate_answer)
builder.add_node("fallback", no_documents)

builder.add_edge(START, "retrieve")

builder.add_conditional_edges(
    "retrieve",
    decide_next_step,
    {
        "generate": "generate",
        "fallback": "fallback"
    }
)

builder.add_edge("generate", END)
builder.add_edge("fallback", END)

# 7. Compile
graph = builder.compile()


# 8. Test both paths
result1 = graph.invoke({
    "question": "What is the hotel reimbursement limit?",
    "documents_found": False,
    "answer": ""
})

result2 = graph.invoke({
    "question": "What is the work from home policy?",
    "documents_found": False,
    "answer": ""
})

print("Result 1:", result1["answer"])
print("Result 2:", result2["answer"])
