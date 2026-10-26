
from dotenv import load_dotenv
from src.vector_store import load_vector_store

load_dotenv()

# Load the existing Chroma collection
vector_store = load_vector_store()

# Access the underlying Chroma collection
collection = vector_store._collection

# Fetch all stored records
records = collection.get(
    include=["documents", "metadatas"]
)

ids = records["ids"]
documents = records["documents"]
metadatas = records["metadatas"]

print(f"Total records: {len(ids)}")

# Group records by identical document content
content_groups = {}

for record_id, document, metadata in zip(ids, documents, metadatas):
    content_groups.setdefault(document, []).append({
        "id": record_id,
        "source": (metadata or {}).get("source", "unknown")
    })

duplicate_groups = {
    content: entries
    for content, entries in content_groups.items()
    if len(entries) > 1
}

print(f"Unique document contents: {len(content_groups)}")
print(f"Duplicate content groups: {len(duplicate_groups)}")

for index, (content, entries) in enumerate(duplicate_groups.items(), start=1):
    print(f"\nDuplicate group {index}")
    print(f"Source: {entries[0]['source']}")
    print(f"Copies: {len(entries)}")
    print(f"IDs: {[entry['id'] for entry in entries]}")
    print(f"Content: {content[:200]}")
