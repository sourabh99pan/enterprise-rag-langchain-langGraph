
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.vector_store import load_vector_store

load_dotenv()

vector_store = load_vector_store()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

question = (
    "Explain annual leave, sick leave, "
    "and carry-forward rules."
)

# Ask the LLM to generate focused search queries.
prompt = f"""
You are helping retrieve information from company policy documents.

Generate exactly 3 focused search queries that together cover
all the important parts of the user's question.

Return only the queries, one per line.
Do not number them or add explanations.

User question: {question}
"""

response = llm.invoke(prompt)

sub_queries = [
    line.strip().lstrip("- ").strip()
    for line in response.content.splitlines()
    if line.strip()
]

print("Generated sub-queries:")
for query in sub_queries:
    print(f"- {query}")

# Retrieve chunks for each generated query.
threshold = 1.0
unique_docs = {}

for query in sub_queries:
    print(f"\nSearching: {query}")

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
            key = (
                doc.page_content,
                str(sorted(doc.metadata.items())),
            )
            unique_docs[key] = doc

# Combine unique chunks.
context = "\n\n".join(
    f"Source: {doc.metadata.get('source', 'unknown')}\n"
    f"Content: {doc.page_content}"
    for doc in unique_docs.values()
)

print(f"\nUnique chunks retained: {len(unique_docs)}")
print("\nCombined context:")
print(context or "No relevant documents found.")
