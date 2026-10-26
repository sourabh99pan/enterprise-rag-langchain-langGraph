
from pathlib import Path

from src.ingestion import load_text_documents
from src.ingestion import chunk_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def test_load_text_documents():
    documents = load_text_documents(str(DOCUMENTS_DIR))

    assert len(documents) == 3

    sources = {doc["metadata"]["source"] for doc in documents}

    # Check that the sources match the expected filenames
    assert sources == {
        "leave_policy.txt",
        "it_security_policy.txt",
        "expense_policy.txt",
    }
    # Check that the content of each document is not empty and that the source ends with ".txt"
    for doc in documents:
        assert doc["content"]
        assert doc["metadata"]["source"].endswith(".txt")

def test_chunk_documents_preserves_source_metadata():
    documents = load_text_documents(str(DOCUMENTS_DIR))
    chunks = chunk_documents(documents)

    assert len(chunks) > 0
    assert all("source" in chunk.metadata for chunk in chunks)

    sources = {chunk.metadata["source"] for chunk in chunks}

    assert sources == {
        "leave_policy.txt",
        "it_security_policy.txt",
        "expense_policy.txt",
    }


def test_chunks_are_not_empty():
    documents = load_text_documents(str(DOCUMENTS_DIR))
    chunks = chunk_documents(documents)

    assert all(chunk.page_content.strip() for chunk in chunks)
