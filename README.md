# RAG Chatbot Project

This project is a retrieval-first chatbot built with LangChain and ChromaDB. It ingests text and PDF documents, chunks and embeds them, and retrieves the most relevant passages for a user query.

The current version intentionally does not call a real LLM yet. The `generate_response()` placeholder is isolated and ready to be connected to an API later once the key and provider are provided.

## What it does

- loads `.txt`, `.md`, and `.pdf` files from the `data/` folder
- splits documents into chunks with configurable overlap
- persists chunk metadata including source and page provenance
- stores chunks in a local Chroma vector database
- retrieves relevant chunks with a configurable similarity threshold
- returns source citations and a future-ready chat history structure
- exposes a FastAPI endpoint and a Streamlit interface

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template and fill in any future keys if needed:

```bash
copy .env.example .env
```

4. Put documents in `data/`.
5. Run ingestion:

```bash
python src/ingest.py
```

6. Start the API:

```bash
uvicorn src.api:app --reload
```

7. Or run the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## Environment variables

The project supports `RAG_*` settings and loads `.env` automatically via `python-dotenv`.

The template includes a placeholder for `GROQ_API_KEY`, but the actual LLM integration is intentionally deferred and not used yet.

## API usage

`GET /health` checks whether the vector store directory exists.

`POST /chat` accepts a JSON payload such as:

```json
{"question": "What is RAG?", "top_k": 3, "similarity_threshold": 0.75}
```

The response includes:

- `question`
- `context`
- `sources`
- `citations`
- `relevant_found`
- `response`
- `chat_history`

## Project structure

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

## Notes

- The application is retrieval-first and intentionally does not include a provider-specific LLM call yet.
- `generate_response()` is the single injection point for the future API call.
- The actual LLM provider, key handling, and runtime model call will be added in a later phase.

## License

MIT
