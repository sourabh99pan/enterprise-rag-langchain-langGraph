
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


def create_vector_store(chunks):
    """Create a persistent vector store from document chunks."""

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not configured.")

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="enterprise_policy_docs",
        persist_directory=str(CHROMA_DIR),
    )

    return vector_store


def load_vector_store():
    """Load the existing persistent Chroma collection."""

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not configured.")

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = Chroma(
        collection_name="enterprise_policy_docs",
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )

    return vector_store
