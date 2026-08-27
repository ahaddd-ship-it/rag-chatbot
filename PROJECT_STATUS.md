# RAG Chatbot - Project Status

## Current Status

The project provides a model-agnostic FastAPI retrieval service. It loads the
persisted Chroma collection, retrieves relevant chunks, and returns context and
source metadata in a response structure ready for future LLM integration.

### 1. Document Ingestion Pipeline (`src/ingest.py`)
- Loads documents from TXT, PDF, or Markdown files
- Splits documents into chunks with configurable size (default: 500) and overlap (default: 50)
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Preserves loader metadata and source provenance
- Uses deterministic IDs and replaces chunks for an already-ingested source
- Persists data in a named Chroma collection

### 2. Vector Store (`chroma_db/`)
- SQLite-based Chroma database created successfully
- Contains 5 embedded document chunks from the sample document
- Supports persistence across sessions

### 3. API (`src/api.py`)
- `GET /health` reports vector-store availability
- `POST /chat` returns `question`, `context`, and `sources`
- Invalid questions return HTTP 422
- Missing or empty vector stores return HTTP 503 with an explicit error

## 📁 Project Structure

```
rag-chatbot/
├── data/
│   └── sample_document.txt          # Sample document for testing
├── src/
│   ├── chatbot.py                    # Retrieval service
│   ├── api.py                         # FastAPI application
│   ├── config.py                     # Configuration settings
│   ├── ingest.py                     # Document ingestion pipeline
│   └── main.py                       # Entry point
├── chroma_db/                        # Persisted vector store (gitignored)
├── requirements.txt                  # Python dependencies
└── PROJECT_STATUS.md                 # This file
```

## 🚀 How to Use

### Option 1: Run Ingestion Script
```bash
python src/ingest.py
```
Ingests documents from `data/sample_document.txt` into the vector store.

### Option 2: Run the API
```bash
uvicorn src.api:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
	-H "Content-Type: application/json" \
	-d "{\"question\":\"What is RAG?\"}"
```

## 📊 Testing Results

| Component | Status | Notes |
|-----------|--------|-------|
| Document Loading | ✅ Working | Successfully loaded sample document |
| Chunking | ✅ Working | Created 5 chunks from document |
| Embeddings | ✅ Working | Using all-MiniLM-L6-v2 model |
| Vector Store | ✅ Working | SQLite database created at `chroma_db/` |
| Context Retrieval | ✅ Working | Found 3 relevant chunks for "What is RAG?" |
| API health | Working | Returns HTTP 200 when the store directory exists |
| API retrieval | Working | Returns context and source metadata |

## 📝 Known Limitations

1. **Network Dependency**: Embedding generation requires internet access on first model download
2. **No LLM generation**: The API intentionally stops at retrieval
3. **No automated test suite**: Verification currently uses import, API, and persistence smoke checks

## 🔄 Next Steps

1. Add automated tests for ingestion, persistence, and retrieval
2. Add more sample documents to the `data/` directory
3. Add an LLM integration behind a separate provider abstraction when required
