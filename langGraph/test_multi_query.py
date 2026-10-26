
from dotenv import load_dotenv
from src.vector_store import load_vector_store

load_dotenv()

vector_store = load_vector_store()

question = (
    "Explain annual leave, sick leave, "
    "and carry-forward rules."
)

sub_queries = [
    "What is the annual leave entitlement for full-time employees?",
    "How many sick leave days are allowed per year?",
    "What are the annual leave carry-forward rules?",
]

threshold = 1.0

# Store unique chunks using their content as the key.
unique_docs = {}

for query in sub_queries:
    print(f"\n{'=' * 15} Query: {query} {'=' * 15}")

    results = vector_store.similarity_search_with_score(
        query,
        k=3,
    )

    for doc, score in results:
        print(
            f"Distance: {score:.4f} | "
            f"Source: {doc.metadata.get('source', 'unknown')}"
        )

        if score <= threshold:
            unique_docs[doc.page_content] = doc

# Combine the unique retrieved chunks.
context = "\n\n".join(
    f"Source: {doc.metadata.get('source', 'unknown')}\n"
    f"Content: {doc.page_content}"
    for doc in unique_docs.values()
)

print("\n" + "=" * 20)
print(f"Original question: {question}")
print(f"Unique chunks retained: {len(unique_docs)}")
print("\nCombined context:")
print(context or "No relevant documents found.")
