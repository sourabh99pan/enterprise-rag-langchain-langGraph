
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = PROJECT_ROOT / "chroma_db"


@pytest.fixture(scope="module")
def vector_store():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return Chroma(
        collection_name="enterprise_policy_docs",
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )


def test_hotel_policy_is_retrieved(vector_store):
    results = vector_store.similarity_search(
        "What is the maximum hotel reimbursement per night?",
        k=3,
    )

    assert any(
        "₹4,000 per night" in doc.page_content
        and doc.metadata.get("source") == "expense_policy.txt"
        for doc in results
    )


def test_password_policy_is_retrieved(vector_store):
    results = vector_store.similarity_search(
        "What are the password requirements?",
        k=3,
    )

    assert any(
        "12 characters" in doc.page_content
        and doc.metadata.get("source") == "it_security_policy.txt"
        for doc in results
    )


def test_leave_policy_is_retrieved(vector_store):
    results = vector_store.similarity_search(
        "How many annual leave days do full-time employees receive?",
        k=3,
    )

    assert any(
        "20 days" in doc.page_content
        and doc.metadata.get("source") == "leave_policy.txt"
        for doc in results
    )
