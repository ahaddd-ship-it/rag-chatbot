from langchain_core.documents import Document
import pytest

from src import config
from src import ingest


class DummyEmbeddings:
    def embed_documents(self, texts):
        return [[0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 0.0]


def test_load_document_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Alpha beta gamma\nDelta epsilon.", encoding="utf-8")

    docs = ingest.load_document(str(file_path))

    assert len(docs) == 1
    assert docs[0].metadata["source"] == str(file_path)


def test_split_documents_preserves_metadata():
    docs = [
        Document(
            page_content="one two three four five six seven eight nine ten",
            metadata={"source": "example.txt", "page": 2},
        )
    ]

    chunks = ingest.split_documents(docs, chunk_size=12, overlap=3)

    assert chunks
    assert all(chunk.metadata["source"] == "example.txt" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)


def test_embed_and_store_preserves_metadata_and_distinct_sources(tmp_path, monkeypatch):
    original_collection = config.CHROMA_COLLECTION
    original_embeddings = ingest.HuggingFaceEmbeddings

    config.CHROMA_COLLECTION = "test_rag_ingest_metadata"
    monkeypatch.setattr(ingest, "HuggingFaceEmbeddings", lambda model_name: DummyEmbeddings())

    try:
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("Alpha beta gamma delta epsilon.", encoding="utf-8")
        file_b.write_text("Omega theta iota kappa lambda.", encoding="utf-8")

        chunks = [
            Document(page_content="alpha chunk", metadata={"source": str(file_a), "page": 1}),
            Document(page_content="beta chunk", metadata={"source": str(file_b), "page": 3}),
        ]

        stored = ingest.embed_and_store(chunks, str(tmp_path / "vectors"))

        assert stored == 2

        from langchain_chroma import Chroma

        vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION,
            persist_directory=str(tmp_path / "vectors"),
            embedding_function=DummyEmbeddings(),
        )
        metadatas = vectorstore.get(include=["metadatas"])["metadatas"]
        sources = {meta["source"] for meta in metadatas}
        pages = {meta["page"] for meta in metadatas}

        assert sources == {str(file_a), str(file_b)}
        assert pages == {1, 3}
    finally:
        config.CHROMA_COLLECTION = original_collection
        ingest.HuggingFaceEmbeddings = original_embeddings


def test_ingest_documents_skips_unchanged_files(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    vector_dir = tmp_path / "vectors"
    data_dir.mkdir()
    (data_dir / "one.txt").write_text("One document.", encoding="utf-8")
    (data_dir / "two.md").write_text("Two document.", encoding="utf-8")

    calls = []

    def fake_embed_and_store(chunks, persist_dir):
        calls.append((chunks, persist_dir))
        return len(chunks)

    monkeypatch.setattr(ingest, "embed_and_store", fake_embed_and_store)

    first = ingest.ingest_documents(data_dir, vector_dir)
    second = ingest.ingest_documents(data_dir, vector_dir)

    assert len(first["ingested"]) == 2
    assert first["skipped"] == []
    assert second["ingested"] == []
    assert len(second["skipped"]) == 2
    assert len(calls) == 2


def test_list_documents_in_folder_rejects_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="Documents folder does not exist"):
        ingest.list_documents_in_folder(tmp_path / "missing")


def test_load_document_rejects_unsupported_file(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("not supported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest.load_document(str(file_path))
