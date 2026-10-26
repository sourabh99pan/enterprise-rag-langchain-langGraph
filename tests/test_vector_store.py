
from pathlib import Path

from dotenv import load_dotenv

from src.ingestion import load_text_documents, chunk_documents
from src.vector_store import create_vector_store


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

load_dotenv(PROJECT_ROOT / ".env")


def test_vector_store_retrieves_relevant_policy():
    documents = load_text_documents(str(DOCUMENTS_DIR))
    chunks = chunk_documents(documents)

    vector_store = create_vector_store(chunks)

    results = vector_store.similarity_search(
        "What is the maximum hotel reimbursement per night?",
        k=3,
    )

    assert len(results) == 3

    assert any(
        "₹4,000 per night" in doc.page_content
        and doc.metadata.get("source") == "expense_policy.txt"
        for doc in results
    )


def test_retrieved_documents_preserve_source():
    documents = load_text_documents(str(DOCUMENTS_DIR))
    chunks = chunk_documents(documents)

    vector_store = create_vector_store(chunks)

    results = vector_store.similarity_search(
        "What are the password requirements?",
        k=3,
    )

    assert len(results) == 3
    assert all(doc.metadata.get("source") for doc in results)
