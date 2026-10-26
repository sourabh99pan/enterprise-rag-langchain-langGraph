from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma(
    collection_name="enterprise_policy_docs",
    embedding_function=embedding_model,
    persist_directory=str(CHROMA_DIR),
)

print("Stored chunks:", vector_store._collection.count())

question = "What is the maximum hotel reimbursement per night?"

results = vector_store.similarity_search(question, k=5)

print("Retrieved results:", len(results))

for rank, doc in enumerate(results, start=1):
    print(f"\n--- Result {rank} ---")
    print("Source:", doc.metadata.get("source"))
    print(doc.page_content)