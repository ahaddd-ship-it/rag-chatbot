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
    top_k: int | None = Field(default=None, ge=1)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    chat_history: list[dict] | None = None


class ChatResponse(BaseModel):
    question: str
    context: str
    sources: list[str]
    citations: list[str]
    relevant_found: bool
    response: str
    chat_history: list[dict]


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
        result = chat(
            request.question,
            config.CHROMA_PERSIST_DIR,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            chat_history=request.chat_history,
        )
        if not result["relevant_found"]:
            return ChatResponse(
                question=request.question,
                context="",
                sources=[],
                citations=[],
                relevant_found=False,
                response=result["response"],
                chat_history=result["chat_history"],
            )
        return ChatResponse(**result)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error