
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# Step 1: Define the state
class GraphState(TypedDict):
    question: str
    answer: str


# Step 2: Create a node
def answer_question(state: GraphState):
    question = state["question"]

    return {
        "answer": f"You asked: {question}"
    }


# Step 3: Create the graph builder
builder = StateGraph(GraphState)

# Step 4: Register the node
builder.add_node("answer_question", answer_question)

# Step 5: Connect the workflow
builder.add_edge(START, "answer_question")
builder.add_edge("answer_question", END)

# Step 6: Compile the graph
graph = builder.compile()

# Step 7: Execute the graph
result = graph.invoke({
    "question": "What is RAG?",
    "answer": ""
})

print(result)
