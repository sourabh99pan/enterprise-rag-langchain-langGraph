
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

from src.vector_store import load_vector_store


# ============================================================
# 1. Initialize shared resources
# ============================================================

vector_store = load_vector_store()

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

def retrieve_documents(state: RAGState):
    """Retrieve relevant policy chunks using multi-query retrieval."""

    question = state["messages"][-1].content

    prompt = f"""
You are helping retrieve information from company policy documents.

Generate exactly 3 focused search queries that together cover
the important parts of the user's question.

Return only the queries, one per line.
Do not number them or add explanations.

User question: {question}
"""

    response = llm.invoke(prompt)

    sub_queries = [
        line.strip().lstrip("- ").strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    print("\nGenerated sub-queries:")
    for query in sub_queries:
        print(f"- {query}")

    attempt = state.get("attempt", 0) + 1
    print(f"\nRetrieval attempt: {attempt}")

    k = 3 if attempt == 1 else 5
    threshold = 1.0 if attempt == 1 else 1.2

    unique_docs = {}

    for query in sub_queries:
        print(f"\nSearching: {query}")

        results = vector_store.similarity_search_with_score(
            query,
            k=k,
        )

        for doc, score in results:
            print(
                f"Distance: {score:.4f} | "
                f"Source: {doc.metadata.get('source', 'unknown')}"
            )

            if score <= threshold:
                key = (
                    doc.page_content,
                    str(sorted(doc.metadata.items())),
                )
                unique_docs[key] = doc

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in unique_docs.values()
    )

    print(f"\nUnique chunks retained: {len(unique_docs)}")
    print("\nFiltered context:")
    print(context or "No relevant documents found.")

    return {
        "context": context,
        "attempt": attempt,
    }


def check_answerability(state: RAGState):
    """Check whether the retrieved context supports an answer."""

    question = state["messages"][-1].content
    context = state.get("context", "")

    prompt = f"""
Determine whether the provided policy context contains
enough evidence to answer the user's question.

Question:
{question}

Policy context:
{context if context else "No context was retrieved."}

Reply with exactly YES or NO.

Reply YES only when the context supports answering
the actual question. Otherwise, reply NO.
"""

    response = llm.invoke(prompt)
    decision = response.content.strip().upper()
    answerable = decision.startswith("YES")

    print(f"\nAnswerability decision: {decision}")
    print(f"Answerable: {answerable}")

    return {"answerable": answerable}


def generate_answer(state: RAGState):
    """Generate an answer using retrieved policy context."""

    context = state.get("context", "")

    system_message = SystemMessage(
        content=f"""
You are an enterprise policy assistant.

Answer using only the policy context below.
If the context does not contain the answer, say:
"I don't know based on the available policy documents."

Mention the source filename when answering.

Policy context:
{context}
"""
    )

    response = llm.invoke(
        [system_message] + state["messages"]
    )

    return {"messages": [response]}


def human_review(state: RAGState):
    """Pause for a human to approve or correct the answer."""

    draft_answer = state["messages"][-1].content

    reviewed_answer = interrupt(
        {
            "message": "Please review the generated policy answer.",
            "draft_answer": draft_answer,
            "instruction": "Enter the approved or corrected answer.",
        }
    )

    return {
        "messages": [
            AIMessage(content=reviewed_answer)
        ]
    }


def route_after_check(state: RAGState):
    """Choose whether to generate, retry, or fall back."""

    if state.get("answerable", False):
        return "generate"

    if state.get("attempt", 0) < 2:
        return "retrieve"

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
# 5. Build the Enterprise RAG graph
# ============================================================

builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve_documents)
builder.add_node("check_answerability", check_answerability)
builder.add_node("generate", generate_answer)
builder.add_node("human_review", human_review)
builder.add_node("fallback", fallback_answer)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "check_answerability")

builder.add_conditional_edges(
    "check_answerability",
    route_after_check,
    {
        "generate": "generate",
        "retrieve": "retrieve",
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

rag_graph = builder.compile(
    checkpointer=InMemorySaver()
)


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

if __name__ == "__main__":

    # Change this to "tool" or "rag" to test automatic calculator calling.
    WORKFLOW = "tool"  # "tool or rag"

    if WORKFLOW == "rag":

        config = {
            "configurable": {
                "thread_id": "policy-chat-001"
            }
        }

        question = input("\nEnter your policy question: ").strip()
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
