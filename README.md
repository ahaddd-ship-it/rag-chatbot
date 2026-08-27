# RAG Chatbot Project

A simple Retrieval-Augmented Generation (RAG) chatbot that ingests documents into a Chroma vector store.

## Features

- Load documents from TXT, PDF, or Markdown files
- Split documents into chunks with configurable size and overlap
- Generate embeddings using sentence-transformers/all-MiniLM-L6-v2
- Persist vector data in ChromaDB with disk persistence enabled

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place a document (TXT, PDF, or MD) in the `data/` directory

3. Run the ingestion script:
   ```bash
   python src/ingest.py
   ```

4. The vector store will be created and persisted in `chroma_db/`

## Project Structure

```
rag-chatbot/
├── data/              # Input documents
├── src/
│   ├── ingest.py      # Document ingestion pipeline
│   └── config.py      # Configuration settings
├── chroma_db/         # Persisted vector store (gitignored)
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## License

MIT
