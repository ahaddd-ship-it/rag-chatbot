"""Runtime configuration for the RAG chatbot."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
