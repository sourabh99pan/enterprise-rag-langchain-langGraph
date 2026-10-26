
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.rag_chain import create_rag_chain

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = PROJECT_ROOT / "chroma_db"


@pytest.fixture(scope="module")
def rag_chain():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma(
        collection_name="enterprise_policy_docs",
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )

    return create_rag_chain(vector_store)


def test_hotel_reimbursement_answer(rag_chain):
    answer = rag_chain.invoke(
        "What is the maximum hotel reimbursement per night?"
    )

    assert "4,000" in answer
    assert "expense_policy.txt" in answer


def test_unknown_policy_question(rag_chain):
    answer = rag_chain.invoke(
        "What is the company's work from home policy?"
    )

    assert "don't know" in answer.lower()


def test_meal_expense_answer(rag_chain):
    answer = rag_chain.invoke(
        "What is the daily meal expense limit?"
    )

    assert "800" in answer
    assert "expense_policy.txt" in answer
 