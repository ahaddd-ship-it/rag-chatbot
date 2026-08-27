"""Document ingestion pipeline for RAG chatbot."""

import os
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from src import config
except ModuleNotFoundError:
    import config


def load_document(filepath: str) -> list:
    """Load document from file.

    Args:
        filepath: Path to the document file.

    Returns:
        List of document pages/sections.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(filepath)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()


def split_documents(documents: list, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split documents into chunks.

    Args:
        documents: List of document pages/sections.
        chunk_size: Size of each chunk.
        overlap: Overlap between chunks.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_documents(documents)


def embed_and_store(chunks: list, persist_dir: str) -> int:
    """Embed chunks and store in Chroma vector store.

    Args:
        chunks: List of text chunks to embed.
        persist_dir: Directory to persist the Chroma database.

    Returns:
        Number of chunks embedded.
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(persist_path),
        embedding_function=embeddings,
    )

    source = str(Path(chunks[0].metadata["source"]).resolve()) if chunks else ""
    if source:
        vectorstore.delete(where={"source": source})

    ids = []
    docs = []
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata["source"] = source
        docs.append(Document(page_content=chunk.page_content, metadata=metadata))
        digest = hashlib.sha256(
            f"{source}:{index}:{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        ids.append(digest)

    if docs:
        vectorstore.add_documents(documents=docs, ids=ids)

    return len(chunks)


def ingest_document(filepath: str, persist_dir: str, chunk_size: int = 500, overlap: int = 50) -> int:
    """Ingest a single document into the vector store.

    Args:
        filepath: Path to the document file.
        persist_dir: Directory to persist the Chroma database.
        chunk_size: Size of each chunk.
        overlap: Overlap between chunks.

    Returns:
        Number of chunks embedded.
    """
    # Load document
    print(f"Loading document from {filepath}...")
    documents = load_document(filepath)
    print(f"Loaded {len(documents)} document(s).")

    # Split into chunks
    print("Splitting documents into chunks...")
    chunks = split_documents(documents, chunk_size=chunk_size, overlap=overlap)
    print(f"Created {len(chunks)} chunks.")

    # Embed and store
    print(f"Embedding and storing in {persist_dir}...")
    for chunk in chunks:
        chunk.metadata.setdefault("source", filepath)
    num_chunks = embed_and_store(chunks, persist_dir)
    print(f"\nSuccessfully embedded {num_chunks} chunks!")

    return num_chunks


if __name__ == "__main__":
    # Default document path
    doc_path = config.DATA_DIR / config.DOC_FILE

    # Ensure data directory exists
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)

    # Run ingestion
    num_chunks = ingest_document(str(doc_path), str(config.CHROMA_PERSIST_DIR), config.CHUNK_SIZE, config.OVERLAP)
