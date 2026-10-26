
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_text_documents(data_dir: str) -> list[dict]:
    """Load TXT files and preserve their source filenames."""
    documents = []
    folder = Path(data_dir)

    if not folder.exists():
        raise FileNotFoundError(
            f"Document directory not found: {folder.resolve()}"
        )

    for file_path in sorted(folder.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Skipping empty file: {file_path.name}")
            continue

        documents.append({
            "content": text,
            "metadata": {
                "source": file_path.name
            }
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[Document]:
    """Convert loaded documents to LangChain Documents and split them."""

    langchain_documents = [
        Document(
            page_content=doc["content"],
            metadata=doc["metadata"]
        )
        for doc in documents
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(langchain_documents)

    return chunks
