"""Model-agnostic RAG retrieval."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from src import config
except ModuleNotFoundError:
    import config


def get_vector_store(persist_dir=None) -> Chroma:
    """Load the persisted vector store."""
    persist_path = Path(config.CHROMA_PERSIST_DIR if persist_dir is None else persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Vector store does not exist at {persist_path}. Run ingestion first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )

    count = vectorstore._collection.count()
    if count == 0:
        raise ValueError(f"Vector store collection '{config.CHROMA_COLLECTION}' is empty.")
    return vectorstore


def generate_response(query: str, context: str, system_prompt: str) -> str:
    """Generate a Gemini response, or use the offline placeholder without a key."""
    if not context.strip():
        return "[No relevant context was found for this query.]"

    if not config.GEMINI_API_KEY:
        return f"[LLM not connected] Retrieved context:\n{context}"

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Gemini support is not installed. Install dependencies from requirements.txt."
        ) from exc

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        interaction = client.interactions.create(
            model=config.GEMINI_MODEL,
            system_instruction=system_prompt,
            input=f"Question: {query}\n\nRetrieved context:\n{context}",
        )
        response = interaction.output_text
    except Exception as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc

    if not response:
        raise RuntimeError("Gemini returned an empty response.")
    return response


def retrieve_context(
    query: str,
    vectorstore: Chroma,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list:
    """Retrieve relevant context while dropping low-similarity matches."""
    if not query.strip():
        raise ValueError("Query must not be empty.")

    resolved_top_k = config.TOP_K if top_k is None else top_k
    resolved_threshold = (
        config.SIMILARITY_THRESHOLD if similarity_threshold is None else similarity_threshold
    )

    if resolved_top_k < 1:
        raise ValueError("top_k must be at least 1.")

    results = vectorstore.similarity_search_with_score(query, k=resolved_top_k)
    relevant_docs = []
    for doc, score in results:
        # Chroma returns distance scores: lower values are more relevant.
        if score <= resolved_threshold:
            doc.metadata["score"] = float(score)
            relevant_docs.append(doc)

    return relevant_docs


def chat(
    query: str,
    persist_dir=None,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    chat_history: list | None = None,
    system_prompt: str | None = None,
) -> dict:
    """Retrieve context, build response metadata, and keep a future-ready chat history."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    vectorstore = get_vector_store(persist_dir)
    docs = retrieve_context(query, vectorstore, top_k=top_k, similarity_threshold=similarity_threshold)

    formatted_context = []
    citations = []
    sources = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = metadata.get("source")
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
        score = metadata.get("score")

        citation = f"[{source or 'unknown'}"
        if page is not None:
            citation += f", page {page}"
        if chunk_index is not None:
            citation += f", chunk {chunk_index}"
        if score is not None:
            citation += f", score {score:.4f}"
        citation += "]"

        citations.append(citation)
        if source:
            sources.append(source)
        formatted_context.append(f"{doc.page_content} {citation}")

    context = "\n\n".join(formatted_context)
    history = list(chat_history or [])
    history.append({"role": "user", "content": query})
    response_text = generate_response(query, context, system_prompt or config.DEFAULT_SYSTEM_PROMPT)
    history.append({"role": "assistant", "content": response_text})

    return {
        "question": query,
        "context": context,
        "sources": list(dict.fromkeys(sources)),
        "citations": citations,
        "relevant_found": bool(docs),
        "response": response_text,
        "chat_history": history,
    }


if __name__ == "__main__":
    print("Run the API with: uvicorn src.api:app --reload")
