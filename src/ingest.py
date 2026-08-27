"""Document ingestion pipeline for RAG chatbot."""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


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
    # Create embeddings using HuggingFace all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    # In newer LangChain versions, chunks are dictionaries with 'page_content' key
    # Extract page contents from chunks
    docs = [Document(page_content=chunk.page_content) for chunk in chunks]

    # Create and persist Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

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
    num_chunks = embed_and_store(chunks, persist_dir)
    print(f"\nSuccessfully embedded {num_chunks} chunks!")

    return num_chunks


if __name__ == "__main__":
    # Default document path
    doc_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_document.txt")

    # Ensure data directory exists
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)

    # Run ingestion
    num_chunks = ingest_document(doc_path, "chroma_db")
