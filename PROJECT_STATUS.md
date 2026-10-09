# RAG Chatbot - Project Status

## Current status

The project includes a retrieval-first RAG pipeline that ingests documents, stores embeddings in Chroma, and returns source-aware context with citations. Google Gemini generation is available when `GEMINI_API_KEY` is configured; otherwise the app remains in offline retrieval mode.

### Implemented

- folder-based document ingestion for `.txt`, `.md`, and `.pdf`
- deterministic per-chunk IDs and metadata preservation
- manifest-based skip logic for unchanged files
- configurable chunk size, overlap, top-k, and similarity threshold
- source citations and retrieval metadata in the response payload
- isolated `generate_response()` with Google Gemini integration and offline fallback
- chat-history structure ready for later LLM calls
- `.env` loading via `python-dotenv`
- Streamlit UI for uploading docs and asking questions
- basic pytest coverage for ingestion and retrieval behavior

### Pending / intentionally deferred

- streaming responses
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
- Gemini calls are isolated from retrieval and covered by a fake-client test

## Next step

Configure `GEMINI_API_KEY` in a local ignored `.env` file to enable Gemini generation. Keep provider-specific changes isolated to `generate_response()`.
