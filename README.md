# RAG Chatbot Project

This project is a retrieval-first chatbot built with LangChain and ChromaDB. It ingests text and PDF documents, chunks and embeds them, and retrieves the most relevant passages for a user query.

Generation uses Google Gemini when `GEMINI_API_KEY` is configured. Without a key, the application remains usable in offline retrieval mode and returns the retrieved context through a clearly labeled fallback.

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

The template includes `GEMINI_API_KEY=your_key_here`. Keep the real key only in the ignored `.env` file or another secret manager.

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

- `generate_response()` is the single provider boundary for Gemini generation.
- Missing keys use offline retrieval mode; provider failures are surfaced as API errors.
- Do not commit `.env` or paste real API keys into source files.

## License

MIT
