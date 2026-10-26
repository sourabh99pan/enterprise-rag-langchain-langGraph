
import json
from pathlib import Path
from dotenv import load_dotenv
from src.vector_store import load_vector_store

load_dotenv()

vector_store = load_vector_store()
collection = vector_store._collection

records = collection.get(include=["documents", "metadatas", "embeddings"])

backup = {
    "ids": records["ids"],
    "documents": records["documents"],
    "metadatas": records["metadatas"],
    "embeddings": (
        records["embeddings"].tolist()
        if records["embeddings"] is not None
        else None
    ),
}

backup_path = Path("chroma_backup.json")
backup_path.write_text(
    json.dumps(backup, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Backed up {len(backup['ids'])} records to {backup_path.resolve()}")
