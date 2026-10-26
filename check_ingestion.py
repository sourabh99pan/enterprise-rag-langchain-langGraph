from pathlib import Path
from src.ingestion import load_text_documents, chunk_documents

documents_dir = Path("data/documents")

documents = load_text_documents(str(documents_dir))
chunks = chunk_documents(documents)

print(f"Total documents: {len(documents)}")
print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(f"Source: {chunk.metadata['source']}")
    print(f"Characters: {len(chunk.page_content)}")
    print(chunk.page_content)