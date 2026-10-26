
import os
from pathlib import Path

from dotenv import load_dotenv

from src.ingestion import load_text_documents, chunk_documents
from src.vector_store import create_vector_store

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

documents_dir = PROJECT_ROOT / "data" / "documents"

documents = load_text_documents(str(documents_dir))
chunks = chunk_documents(documents)

vector_store = create_vector_store(chunks)

print("Documents loaded:", len(documents))
print("Chunks created:", len(chunks))
print("Vector store created successfully!")

question = "What is the maximum hotel reimbursement per night?"

results = vector_store.similarity_search(question, k=3)

for rank, doc in enumerate(results, start=1):
    print(f"\n--- Result {rank} ---")
    print("Source:", doc.metadata.get("source"))
    print(doc.page_content)
