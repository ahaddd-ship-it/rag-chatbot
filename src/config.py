"""Configuration for RAG chatbot."""

# Document path
DATA_DIR = "data"
DOC_FILE = "sample_document.txt"

# Chunking settings
CHUNK_SIZE = 500
OVERLAP = 50

# Chroma database
CHROMA_PERSIST_DIR = "chroma_db"

# LLM Configuration (optional - can be set via environment variables)
HUGGING_FACE_API_KEY = ""  # Set this for Hugging Face Inference API
OLLAMA_HOST = "http://localhost:11434"  # Ollama server URL
OLLAMA_MODEL = "mistral:7b"  # Ollama model to use

# For testing without LLM, set USE_LOCAL_MODELL True
USE_LOCAL_MODEL = False  # Set to True if you have a local model available
