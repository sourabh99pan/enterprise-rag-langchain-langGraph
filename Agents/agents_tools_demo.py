 
from pathlib import Path
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# 1. Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# 2. Load the existing Chroma vector store
vector_store = Chroma(
    collection_name="enterprise_policy_docs",
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-3-small"
    ),
    persist_directory=str(PROJECT_ROOT / "chroma_db"),
)


# 3. Define the calculator tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    result = a * b
    logger.info("multiply result: %s", result)
    return result


# 4. Define the policy-search tool
@tool
def search_policy(question: str) -> str:
    """Search company policy documents for relevant information."""
    logger.info("Searching policy documents")

    docs = vector_store.similarity_search(question, k=3)

    logger.info("Policy search retrieved %s chunks", len(docs))

    if not docs:
        return "No relevant policy documents were retrieved."

    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )


# 5. Bind both tools to the LLM
tools = [multiply, search_policy]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)


# 6. Define the LLM node
def call_llm(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])

    if response.tool_calls:
        for tool_call in response.tool_calls:
            logger.info(
                "Tool selected: %s | Arguments: %s",
                tool_call["name"],
                tool_call["args"],
            )
    else:
        logger.info("LLM returned a final response")

    return {"messages": [response]}


# 7. Build the LangGraph agent workflow
builder = StateGraph(MessagesState)

builder.add_node("call_llm", call_llm)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "call_llm")

builder.add_conditional_edges(
    "call_llm",
    tools_condition,
)

builder.add_edge("tools", "call_llm")

agent = builder.compile()


# 8. Run the agent
if __name__ == "__main__":
    question = input("Ask a calculation or policy question: ").strip()

    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })

    print("\nFinal answer:")
    print(result["messages"][-1].content)
