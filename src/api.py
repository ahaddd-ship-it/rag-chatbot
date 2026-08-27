"""FastAPI service for model-agnostic RAG retrieval."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from src import config
    from src.chatbot import chat
except ModuleNotFoundError:
    import config
    from chatbot import chat


app = FastAPI(title="RAG Chatbot API", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)


class ChatResponse(BaseModel):
    question: str
    context: str
    sources: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    """Return service availability without loading the embedding model."""
    if not config.CHROMA_PERSIST_DIR.exists():
        return {"status": "degraded", "detail": "Vector store is not initialized."}
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Retrieve context and source metadata for a question."""
    try:
        return chat(request.question, config.CHROMA_PERSIST_DIR)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error