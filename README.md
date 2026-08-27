# RAG Chatbot Project

A simple Retrieval-Augmented Generation (RAG) chatbot that ingests documents into a Chroma vector store.

## Features

- Load documents from TXT, PDF, or Markdown files
- Split documents into chunks with configurable size and overlap
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Persist vector data in ChromaDB with deterministic, source-based chunk IDs
- Retrieve context through a model-agnostic API ready for future LLM integration

## Setup

1. Install dependencies in a virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

2. Place a document (TXT, PDF, or MD) in the `data/` directory

3. Run the ingestion script:
   ```bash
   python src/ingest.py
   ```

4. Start the API:
   ```bash
   uvicorn src.api:app --reload
   ```

The vector store is persisted in `chroma_db/`. Re-ingesting the same source replaces
its existing chunks instead of duplicating them. Runtime settings can be overridden
with `RAG_*` environment variables.

## API

`GET /health` reports whether the service has a persisted vector-store directory.

`POST /chat` accepts a JSON body and returns retrieved context and source metadata:

```json
{"question": "What is RAG?"}
```

```json
{"question": "What is RAG?", "context": "...", "sources": [".../sample_document.txt"]}
```

## Project Structure

```
rag-chatbot/
├── data/              # Input documents
├── src/
│   ├── ingest.py      # Document ingestion pipeline
│   ├── chatbot.py     # Retrieval service
│   ├── api.py         # FastAPI application
│   ├── main.py        # Optional CLI retrieval entry point
│   └── config.py      # Environment-backed settings
├── chroma_db/         # Persisted vector store (gitignored)
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## License

MIT
