"""Basic retrieval evaluation for the RAG chatbot."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.chatbot import get_vector_store, retrieve_context


FIXED_QUESTIONS = [
    {
        "query": "What is RAG?",
        "expected_keywords": ["RAG", "retrieval", "generation"],
    },
    {
        "query": "What does the sample document say about ChromaDB?",
        "expected_keywords": ["Chroma", "vector store"],
    },
]


def evaluate() -> list[dict]:
    """Check whether the expected keywords are present in the retrieved context."""
    vectorstore = get_vector_store(config.CHROMA_PERSIST_DIR)
    results = []

    for question in FIXED_QUESTIONS:
        docs = retrieve_context(question["query"], vectorstore, top_k=3, similarity_threshold=0.75)
        content = "\n".join(doc.page_content.lower() for doc in docs)
        matched = sum(1 for keyword in question["expected_keywords"] if keyword.lower() in content)
        results.append(
            {
                "query": question["query"],
                "matched_keywords": matched,
                "expected_keywords": len(question["expected_keywords"]),
                "retrieved_chunks": len(docs),
            }
        )

    return results


if __name__ == "__main__":
    summary = evaluate()
    for item in summary:
        print(item)
