
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

def chatbot(state: MessagesState):
    system_message = SystemMessage(
        content=(
            "You are a helpful enterprise policy assistant. "
            "Answer clearly and concisely. "
            "Do not invent company policy information. "
            "If you do not know the answer, say so."
        )
    )

    response = llm.invoke(
        [system_message] + state["messages"]
    )

    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

config = {
    "configurable": {
        "thread_id": "employee-001"
    }
}

# First question
result1 = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is annual leave?"
            )
        ]
    },
    config=config
)

print("Assistant:", result1["messages"][-1].content)

# Follow-up question in the same conversation
result2 = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Can you explain that more simply?"
            )
        ]
    },
    config=config
)

print("\nAssistant:", result2["messages"][-1].content)

# Inspect conversation history
print("\nConversation history:")
for message in graph.get_state(config).values["messages"]:
    print(f"{message.type}: {message.content}")
