from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict):
    question: str
    leave_result: str
    expense_result: str
    final_answer: str


def check_leave_policy(state: GraphState):
    print("Checking leave policy...")
    return {
        "leave_result": "Annual leave: 20 days per year."
    }


def check_expense_policy(state: GraphState):
    print("Checking expense policy...")
    return {
        "expense_result": "Hotel reimbursement: ₹4,000 per night."
    }


def combine_results(state: GraphState):
    answer = (
        state["leave_result"]
        + "\n"
        + state["expense_result"]
    )
    return {"final_answer": answer}


builder = StateGraph(GraphState)

builder.add_node("leave", check_leave_policy)
builder.add_node("expense", check_expense_policy)
builder.add_node("combine", combine_results)

builder.add_edge(START, "leave")
builder.add_edge(START, "expense")

builder.add_edge("leave", "combine")
builder.add_edge("expense", "combine")

builder.add_edge("combine", END)

graph = builder.compile()

result = graph.invoke({
    "question": "Tell me about leave and hotel reimbursement.",
    "leave_result": "",
    "expense_result": "",
    "final_answer": ""
})

print("\nFinal answer:")
print(result["final_answer"])