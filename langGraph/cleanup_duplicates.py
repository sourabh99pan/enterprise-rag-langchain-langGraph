
import json
from dotenv import load_dotenv
from src.vector_store import load_vector_store

load_dotenv()

vector_store = load_vector_store()
collection = vector_store._collection

records = collection.get(include=["documents", "metadatas"])

# Group records by document content and metadata.
groups = {}

for record_id, document, metadata in zip(
    records["ids"],
    records["documents"],
    records["metadatas"],
):
    key = (
        document,
        json.dumps(metadata or {}, sort_keys=True),
    )
    groups.setdefault(key, []).append(record_id)

# Keep one record per group; mark the remaining IDs for deletion.
duplicate_ids = []

for ids in groups.values():
    duplicate_ids.extend(ids[1:])

print(f"Records before cleanup: {collection.count()}")
print(f"Unique document groups: {len(groups)}")
print(f"Duplicate records to delete: {len(duplicate_ids)}")

if not duplicate_ids:
    print("No duplicates found.")
else:
    confirm = input(
        "Type DELETE DUPLICATES to remove them: "
    )

    if confirm == "DELETE DUPLICATES":
        collection.delete(ids=duplicate_ids)
        print("Duplicate records deleted.")
        print(f"Records remaining: {collection.count()}")
    else:
        print("Cancelled. No records were deleted.")
