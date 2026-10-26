import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict):
    question: str
    results: Annotated[list[str], operator.add]


def check_leave_policy(state: GraphState):
    print("Checking leave policy...")
    return {"results": ["Annual leave: 20 days per year."]}


def check_expense_policy(state: GraphState):
    print("Checking expense policy...")
    return {"results": ["Hotel reimbursement: ₹4,000 per night."]}


def combine_results(state: GraphState):
    answer = "\n".join(state["results"])
    print("\nCombined results:")
    print(answer)
    return {}


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
    "results": []
})

print("\nFinal state:")
print(result)