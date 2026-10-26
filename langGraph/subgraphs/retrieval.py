
import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

class RetrievalState(TypedDict):
    messages: list
    context: str
    answerable: bool
    attempt: int


def generate_search_queries(question: str, llm: Any) -> list[str]:
    """Generate focused search queries."""
    prompt = f"""
Generate exactly 3 focused search queries for this question.
Return only the queries, one per line.

Question: {question}
"""
    response = llm.invoke(prompt)

    return [
        line.strip().lstrip("- ").strip()
        for line in response.content.splitlines()
        if line.strip()
    ]


def get_retrieval_settings(attempt: int) -> tuple[int, float]:
    """Return result count and distance threshold for an attempt."""
    if attempt == 1:
        return 3, 1.0

    return 5, 1.2


def search_unique_documents(
    queries: list[str],
    vector_store: Any,
    k: int,
    threshold: float,
) -> dict:
    """Search queries and retain qualifying unique chunks."""
    unique_docs = {}

    for query in queries:
        results = vector_store.similarity_search_with_score(query, k=k)
        add_qualifying_documents(results, threshold, unique_docs)

    return unique_docs


def add_qualifying_documents(results, threshold: float, unique_docs: dict):
    """Add qualifying documents while avoiding duplicates."""
    for doc, score in results:
        if score <= threshold:
            key = (
                doc.page_content,
                str(sorted(doc.metadata.items())),
            )
            unique_docs[key] = doc


def format_context(unique_docs: dict) -> str:
    """Format retrieved documents into context."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in unique_docs.values()
    )



def retrieve_documents(
    state: RetrievalState,
    vector_store: Any,
    llm: Any,
):
    question = state["messages"][-1].content
    attempt = state.get("attempt", 0) + 1

    logger.info("Retrieval attempt %s started", attempt)

    try:
        queries = generate_search_queries(question, llm)
        if not queries:
            raise ValueError("The LLM generated no search queries.")

        k, threshold = get_retrieval_settings(attempt)

        unique_docs = search_unique_documents(
            queries,
            vector_store,
            k,
            threshold,
        )

        context = format_context(unique_docs)

        logger.info(
            "Retrieval attempt %s completed: %s unique chunks retained",
            attempt,
            len(unique_docs),
        )

        return {
            "context": context,
            "attempt": attempt,
        }

    except Exception:
        logger.exception(
            "Retrieval attempt %s failed",
            attempt,
        )
        raise




def check_answerability(state: RetrievalState, llm: Any):
    question = state["messages"][-1].content
    context = state.get("context", "")

    prompt = f"""
Does the following context contain enough evidence
to answer the question? Reply exactly YES or NO.

Question: {question}

Context:
{context if context else "No context retrieved."}
"""

    try:
        response = llm.invoke(prompt)
        decision = response.content.strip().upper()
        answerable = decision.startswith("YES")

        logger.info(
            "Answerability decision: %s",
            "YES" if answerable else "NO",
        )

        return {"answerable": answerable}

    except Exception:
        logger.exception("Answerability check failed")
        raise



def route_after_check(state: RetrievalState) -> str:
    """Retry once if the context is insufficient."""
    if state.get("answerable", False):
        return "done"

    return "retry" if state.get("attempt", 0) < 2 else "done"


def build_retrieval_subgraph(vector_store: Any, llm: Any):
    """Build and compile the retrieval subgraph."""
    builder = StateGraph(RetrievalState)

    builder.add_node(
        "retrieve",
        lambda state: retrieve_documents(state, vector_store, llm),
    )
    builder.add_node(
        "check_answerability",
        lambda state: check_answerability(state, llm),
    )

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "check_answerability")

    builder.add_conditional_edges(
        "check_answerability",
        route_after_check,
        {"retry": "retrieve", "done": END},
    )

    return builder.compile()
