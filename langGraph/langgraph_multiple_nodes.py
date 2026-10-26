
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define the shared state
class GraphState(TypedDict):
    question: str
    context: str
    answer: str


# 2. Node 1: Retrieve information
def retrieve_information(state: GraphState):
    question = state["question"]

    # Simulate retrieval for learning
    if "hotel" in question.lower():
        context = "Domestic hotel reimbursement is up to ₹4,000 per night."
    else:
        context = "No relevant policy information found."

    return {"context": context}


# 3. Node 2: Generate an answer
def generate_answer(state: GraphState):
    context = state["context"]

    if context == "No relevant policy information found.":
        answer = "I don't know based on the available policy documents."
    else:
        answer = f"Based on the policy: {context}"

    return {"answer": answer}


# 4. Build the graph
builder = StateGraph(GraphState)

builder.add_node("retrieve", retrieve_information)
builder.add_node("generate", generate_answer)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

# 5. Compile the graph
graph = builder.compile()

# 6. Run it
result = graph.invoke({
    "question": "What is the hotel reimbursement limit?",
    "context": "",
    "answer": ""
})

print(result)
 