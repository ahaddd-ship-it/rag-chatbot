import pytest
from langchain_core.documents import Document

from src import config
from src.chatbot import generate_response, retrieve_context


class DummyVectorStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search_with_score(self, query, k):
        return [(doc, 0.2) for doc in self.docs[:1]] + [(doc, 0.9) for doc in self.docs[1:k]]


def test_generate_response_placeholder():
    response = generate_response("What is RAG?", "Some retrieved text.", "System prompt")

    assert response.startswith("[LLM not yet connected]")
    assert "Retrieved context" in response
    assert "Some retrieved text." in response


def test_retrieve_context_filters_low_similarity():
    docs = [
        Document(page_content="important answer", metadata={"source": "a.txt", "page": 1}),
        Document(page_content="irrelevant answer", metadata={"source": "b.txt", "page": 2}),
    ]
    store = DummyVectorStore(docs)

    results = retrieve_context(
        "What is RAG?",
        store,
        top_k=2,
        similarity_threshold=0.5,
    )

    assert len(results) == 1
    assert results[0].page_content == "important answer"


def test_retrieve_context_rejects_empty_query():
    with pytest.raises(ValueError):
        retrieve_context("   ", DummyVectorStore([]), top_k=1, similarity_threshold=0.5)
