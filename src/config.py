"""Runtime configuration for the RAG chatbot."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", PROJECT_ROOT / "data"))
DOC_FILE = os.getenv("RAG_DOC_FILE", "sample_document.txt")
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
CHROMA_PERSIST_DIR = Path(os.getenv("RAG_CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
CHROMA_COLLECTION = os.getenv("RAG_CHROMA_COLLECTION", "rag_documents")
EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.75"))
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "RAG_SYSTEM_PROMPT",
    "You are a helpful assistant. Answer using the retrieved source context only.",
)
