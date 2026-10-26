
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class ApprovalState(TypedDict):
    action: str
    approved: bool


def propose_action(state: ApprovalState):
    print(f"Proposed action: {state['action']}")
    return {}


def human_approval(state: ApprovalState):
    decision = interrupt(
        {
            "message": "Do you approve this action?",
            "action": state["action"],
        }
    )

    return {"approved": decision}


def execute_action(state: ApprovalState):
    if state["approved"]:
        print("Action approved. Executing action...")
    else:
        print("Action rejected. Not executing.")

    return {}


builder = StateGraph(ApprovalState)

builder.add_node("propose_action", propose_action)
builder.add_node("human_approval", human_approval)
builder.add_node("execute_action", execute_action)

builder.add_edge(START, "propose_action")
builder.add_edge("propose_action", "human_approval")
builder.add_edge("human_approval", "execute_action")
builder.add_edge("execute_action", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


config = {
    "configurable": {
        "thread_id": "approval-demo-001"
    }
}

result = graph.invoke(
    {
        "action": "Approve an employee's leave request",
        "approved": False,
    },
    config=config,
)


print("\nGraph paused for human approval.")
print("Interrupt information:", result["__interrupt__"])

# Human reviews the proposed action
decision = input("\nApprove this action? (yes/no): ").strip().lower()

# Resume the same paused graph
result = graph.invoke(
    Command(resume=(decision == "yes")),
    config=config,
)

print("\nFinal state:", result)

