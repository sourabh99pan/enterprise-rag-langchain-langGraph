
from dotenv import load_dotenv
from src.vector_store import load_vector_store

load_dotenv()

vector_store = load_vector_store()

question = "Explain annual leave, sick leave and carry-forward rules."
threshold = 1.0

for k in [1, 3, 5]:
    print(f"\n{'=' * 20} k={k} {'=' * 20}")

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
    )

    relevant_results = [
        (doc, score)
        for doc, score in results
        if score <= threshold
    ]

    for doc, score in results:
        status = "KEPT" if score <= threshold else "FILTERED"
        print(
            f"\nDistance: {score:.4f} | {status}"
            f"\nContent: {doc.page_content[:180]}"
        )

    print(
        f"\nRetrieved: {len(results)}"
        f" | Passed threshold: {len(relevant_results)}"
    )
