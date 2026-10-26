
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI


def format_docs(docs):
    """Combine retrieved document chunks into context."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )


def create_rag_chain_with_context(vector_store):
    """Return the answer and the exact context used to generate it."""

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an enterprise policy assistant.

Answer questions using only the provided context.
If the context does not contain the answer, say:
"I don't know based on the available policy documents."

Do not invent policy information.
Mention the source filename when answering.

Context:
{context}"""
        ),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    answer_chain = prompt | llm | StrOutputParser()

    return (
        {
            "question": RunnablePassthrough(),
            "docs": retriever,
        }
        | RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda state: format_docs(state["docs"])
            )
        )
        | RunnablePassthrough.assign(
            answer=answer_chain
        )
        | RunnableLambda(
            lambda state: {
                "answer": state["answer"],
                "context": state["context"],
                "source_documents": state["docs"],
            }
        )
    )


def create_rag_chain(vector_store):
    """Create the original answer-only RAG chain."""

    return (
        create_rag_chain_with_context(vector_store)
        | RunnableLambda(lambda result: result["answer"])
    )
