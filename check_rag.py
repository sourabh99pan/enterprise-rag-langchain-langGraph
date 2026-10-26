
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.rag_chain import create_rag_chain

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="enterprise_policy_docs",
    embedding_function=embedding_model,
    persist_directory=str(PROJECT_ROOT / "chroma_db")
)

rag_chain = create_rag_chain(vector_store)

question = "What is the company's work from home policy?"

answer = rag_chain.invoke(question)

print("Question:", question)
print("\nRAG Answer:", answer)
