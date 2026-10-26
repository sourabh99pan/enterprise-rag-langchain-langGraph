
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.rag_chain import create_rag_chain_with_context

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

vector_store = Chroma(
    collection_name="enterprise_policy_docs",
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-3-small"
    ),
    persist_directory=str(PROJECT_ROOT / "chroma_db"),
)

rag_chain = create_rag_chain_with_context(vector_store)

result = rag_chain.invoke(
    "What is the maximum hotel reimbursement per night?"
)

print("\nANSWER:")
print(result["answer"])

print("\nACTUAL RETRIEVAL CONTEXT:")
print(result["context"])

print("\nRETRIEVED SOURCES:")
for doc in result["source_documents"]:
    print(doc.metadata.get("source", "unknown"))
