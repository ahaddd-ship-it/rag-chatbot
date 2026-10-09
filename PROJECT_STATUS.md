# RAG Chatbot - Project Status

## Current status

The project now includes a retrieval-first RAG pipeline that ingests documents, stores embeddings in Chroma, and returns source-aware context with citations. The generation layer is intentionally placeholder-based and not connected to a live LLM provider yet.

### Implemented

- folder-based document ingestion for `.txt`, `.md`, and `.pdf`
- deterministic per-chunk IDs and metadata preservation
- manifest-based skip logic for unchanged files
- configurable chunk size, overlap, top-k, and similarity threshold
- source citations and retrieval metadata in the response payload
- placeholder `generate_response()` for future provider integration
- chat-history structure ready for later LLM calls
- `.env` loading via `python-dotenv`
- Streamlit UI for uploading docs and asking questions
- basic pytest coverage for ingestion and retrieval behavior

### Pending / intentionally deferred

- real LLM provider connection
- API key validation and provider-specific handling
- provider-specific response generation and streaming
- broader production hardening beyond the current placeholder flow

## Files

```text
rag-chatbot/
├── data/
├── src/
│   ├── api.py
│   ├── chatbot.py
│   ├── config.py
│   ├── ingest.py
│   ├── main.py
│   └── __init__.py
├── tests/
│   ├── test_ingest.py
│   └── test_chatbot.py
├── .env.example
├── .gitignore
├── chroma_db/
├── requirements.txt
├── README.md
├── PROJECT_STATUS.md
├── streamlit_app.py
├── eval_rag.py
└── .env
```

## Validation notes

- ingestion metadata tests pass
- retrieval threshold logic is in place and can be exercised by the project tests
- the placeholder response remains isolated from any provider-specific code

## Next step

Add a real LLM provider behind `generate_response()` once the API key and target model are ready, keeping all provider-specific work isolated to that function.
