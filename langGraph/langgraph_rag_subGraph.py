
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import sqlite3
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

from src.vector_store import load_vector_store
from langGraph.subgraphs.retrieval import build_retrieval_subgraph
import logging
from logging_config import setup_logging


# ============================================================
# 1. Initialize shared resources
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DB = PROJECT_ROOT / "langgraph_checkpoints.sqlite"

vector_store = load_vector_store()

logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# ============================================================
# 2. Calculator tool and tool-calling LLM
# ============================================================

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


llm_with_tools = llm.bind_tools([multiply])


# ============================================================
# 3. Enterprise RAG state
# ============================================================

class RAGState(MessagesState):
    context: str
    answerable: bool
    attempt: int
    needs_review: bool


# ============================================================
# 4. Enterprise RAG nodes
# ============================================================


def generate_answer(state):
    logger.info("Answer generation started")

    try:
        # Keep your existing SystemMessage and prompt here.
        system_message = SystemMessage(
            content=(
                "Answer the user's question using only the "
                "provided policy context. If the context does "
                "not contain the answer, say you don't know. "
                "Mention the source filename."
                "\n\nContext:\n" + state.get("context", "")
            )
        )

        response = llm.invoke(
            [system_message] + state["messages"]
        )

        logger.info("Answer generation completed")
        return {"messages": [response]}

    except Exception:
        logger.exception("Answer generation failed")
        raise




def human_review(state):
    logger.info("Human review required")

    try:
        reviewed_answer = interrupt({
            "message": "Please review the draft answer.",
            "draft_answer": state["messages"][-1].content,
            "instruction": (
                "Approve or correct the draft answer."
            ),
        })

        if not isinstance(reviewed_answer, str) or not reviewed_answer.strip():
            raise ValueError(
                "Human review returned an empty or invalid answer."
            )

        logger.info("Human review completed")
        return {
            "messages": [
                AIMessage(content=reviewed_answer.strip())
            ]
        }

    except Exception:
        logger.exception("Human-review processing failed")
        raise




def route_inside_retrieval(state: RAGState):
    """Retry retrieval if the context is insufficient."""
    if state.get("answerable", False):
        return "done"

    if state.get("attempt", 0) < 2:
        return "retry"

    return "done"


def route_after_retrieval(state: RAGState):
    """Choose answer generation or fallback after retrieval."""
    if state.get("answerable", False):
        return "generate"

    return "fallback"



def route_after_generation(state: RAGState):
    """Decide whether human review is required."""

    if state.get("needs_review", False):
        return "review"

    return "finish"


def fallback_answer(state: RAGState):
    """Return a safe response when evidence is insufficient."""

    return {
        "messages": [
            AIMessage(
                content=(
                    "I don't know based on the available "
                    "policy documents."
                )
            )
        ]
    }


# ============================================================
#   Build the Enterprise RAG graph
# ============================================================


# ============================================================
# 5. Build the Retrieval Subgraph
# ============================================================

retrieval_subgraph = build_retrieval_subgraph(
    vector_store=vector_store,
    llm=llm,
)


# ============================================================
# 6. Build the Parent Enterprise RAG Graph
# ============================================================

builder = StateGraph(RAGState)

# The compiled retrieval subgraph becomes one parent node.
builder.add_node("retrieval_subgraph", retrieval_subgraph)

builder.add_node("generate", generate_answer)
builder.add_node("human_review", human_review)
builder.add_node("fallback", fallback_answer)

builder.add_edge(START, "retrieval_subgraph")

builder.add_conditional_edges(
    "retrieval_subgraph",
    route_after_retrieval,
    {
        "generate": "generate",
        "fallback": "fallback",
    },
)

builder.add_conditional_edges(
    "generate",
    route_after_generation,
    {
        "review": "human_review",
        "finish": END,
    },
)

builder.add_edge("human_review", END)
builder.add_edge("fallback", END)

checkpoint_conn = sqlite3.connect(
    str(CHECKPOINT_DB),
    check_same_thread=False,
)

checkpointer = SqliteSaver(checkpoint_conn)
rag_graph = builder.compile(checkpointer=checkpointer)

# ============================================================
# 6. Calculator tool-calling node
# ============================================================

def call_llm(state: MessagesState):
    """Let the LLM respond or request a tool."""

    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


# ============================================================
# 7. Build the automatic tool-calling graph
# ============================================================

tool_builder = StateGraph(MessagesState)

tool_builder.add_node("call_llm", call_llm)
tool_builder.add_node("tools", ToolNode([multiply]))

tool_builder.add_edge(START, "call_llm")

tool_builder.add_conditional_edges(
    "call_llm",
    tools_condition,
)

tool_builder.add_edge("tools", "call_llm")

tool_graph = tool_builder.compile()


# ============================================================
# 8. Run the selected workflow
# ============================================================


# Change this to "tool" or "rag" to test automatic calculator calling.
WORKFLOW = "rag"  # "tool" or "rag"
setup_logging()
logger = logging.getLogger(__name__)

logger.info("Enterprise RAG application started")

if __name__ == "__main__":

    if WORKFLOW == "rag":

        config = {
            "configurable": {
                "thread_id": "policy-chat-001"
            }
        }

        action = input(
            "\nChoose: 1 = New question, 2 = Resume pending review: "
        ).strip()

        if action == "1":

            question = input(
                "\nEnter your policy question: "
            ).strip()

            needs_review = input(
                "Require human review? (yes/no): "
            ).strip().lower() == "yes"

            result = rag_graph.invoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "context": "",
                    "attempt": 0,
                    "answerable": False,
                    "needs_review": needs_review,
                },
                config=config,
            )

        elif action == "2":

            reviewed_answer = input(
                "\nEnter the approved or corrected answer: "
            ).strip()

            result = rag_graph.invoke(
                Command(resume=reviewed_answer),
                config=config,
            )

        else:
            raise ValueError("Invalid choice. Enter 1 or 2.")

        if result.get("__interrupt__"):
            interrupt_info = result["__interrupt__"][0].value

            print("\nHuman review required.")
            print("Draft answer:", interrupt_info["draft_answer"])

            reviewed_answer = input(
                "\nEnter the approved or corrected answer: "
            ).strip()

            result = rag_graph.invoke(
                Command(resume=reviewed_answer),
                config=config,
            )

        print("\nFinal answer:")
        print(result["messages"][-1].content)

    elif WORKFLOW == "tool":

        tool_result = tool_graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="What is 4000 multiplied by 3?"
                    )
                ]
            }
        )

        print("\nFinal LLM response:")
        print(tool_result["messages"][-1].content)


    