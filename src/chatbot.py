"""Model-agnostic RAG retrieval."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from src import config
except ModuleNotFoundError:
    import config


def get_vector_store(persist_dir=None) -> Chroma:
    """Load the persisted vector store.

    Args:
        persist_dir: Directory where the vector store is persisted.

    Returns:
        Chroma vector store instance.
    """
    persist_path = Path(config.CHROMA_PERSIST_DIR if persist_dir is None else persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Vector store does not exist at {persist_path}. Run ingestion first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    print(f"Loading vector store from {persist_path}...")
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )
    if vectorstore._collection.count() == 0:
        raise ValueError(f"Vector store collection '{config.CHROMA_COLLECTION}' is empty.")
    print("Vector store loaded successfully.")
    return vectorstore


def retrieve_context(query: str, vectorstore: Chroma, top_k: int = 3) -> list:
    """Retrieve relevant context from the vector store.

    Args:
        query: User's search query.
        vectorstore: Chroma vector store instance.
        top_k: Number of chunks to retrieve.

    Returns:
        List of retrieved document chunks.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    docs = vectorstore.similarity_search(query, k=top_k)
    return docs


def chat(query: str, persist_dir=None) -> dict:
    """Retrieve context and sources for a question.

    Args:
        query: User's question.
        persist_dir: Directory where the vector store is persisted.

    Returns:
        Retrieval result ready for a future LLM response field.
    """
    # Get vector store
    vectorstore = get_vector_store(persist_dir)

    # Retrieve relevant context
    docs = retrieve_context(query, vectorstore, config.TOP_K)

    return {
        "question": query,
        "context": "\n\n".join(doc.page_content for doc in docs),
        "sources": [doc.metadata.get("source") for doc in docs if doc.metadata.get("source")],
    }


if __name__ == "__main__":
    # Default persist directory
    print("Run the API with: uvicorn src.api:app --reload")
