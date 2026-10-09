"""Document ingestion pipeline for RAG chatbot."""

import hashlib
import json
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

try:
    from src import config
except ModuleNotFoundError:
    import config

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _normalize_source_path(filepath: str | Path) -> str:
    """Return a stable absolute path for stored metadata."""
    return str(Path(filepath).resolve())


def _content_hash(filepath: str | Path) -> str:
    """Compute a stable hash for a file to support skip logic."""
    digest = hashlib.sha256()
    with Path(filepath).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_chunk_id(source: str, chunk_index: int, page: int | str | None, content: str) -> str:
    """Build a deterministic chunk ID that remains stable across re-runs."""
    page_value = page if page is not None else "unknown"
    payload = f"{source}:{page_value}:{chunk_index}:{content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_document(filepath: str) -> list:
    """Load a document from disk and preserve source metadata."""
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    try:
        if ext == ".pdf":
            reader = PdfReader(str(path))
            documents = [
                Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": str(path), "page": page_number},
                )
                for page_number, page in enumerate(reader.pages, start=1)
            ]
        else:
            documents = [
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": str(path)},
                )
            ]
    except Exception as exc:  # pragma: no cover - exercised via integration tests
        raise ValueError(f"Failed to read document '{path}': {exc}") from exc

    if not documents:
        raise ValueError(f"No content could be extracted from '{path}'.")

    normalized_source = _normalize_source_path(path)
    for document in documents:
        document.metadata["source"] = normalized_source
        if "page" not in document.metadata:
            document.metadata["page"] = 1

    return documents


def split_documents(documents: list, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split documents into chunks while preserving their provenance metadata."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata.setdefault("source", _normalize_source_path(documents[0].metadata.get("source", "unknown")))
        chunk.metadata.setdefault("page", documents[0].metadata.get("page", 1))
        chunk.metadata["chunk_index"] = index

    return chunks


def embed_and_store(chunks: list, persist_dir: str) -> int:
    """Embed chunks and store them in Chroma without overwriting original metadata."""
    if not chunks:
        return 0

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(persist_path),
        embedding_function=embeddings,
    )

    unique_sources = {
        _normalize_source_path(chunk.metadata.get("source", ""))
        for chunk in chunks
        if chunk.metadata.get("source")
    }
    for source in unique_sources:
        vectorstore.delete(where={"source": source})

    docs = []
    ids = []
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        source = metadata.get("source") or str(Path.cwd())
        metadata["source"] = _normalize_source_path(source)
        metadata["chunk_index"] = metadata.get("chunk_index", index)
        docs.append(Document(page_content=chunk.page_content, metadata=metadata))
        ids.append(
            _stable_chunk_id(
                metadata["source"],
                metadata.get("chunk_index", index),
                metadata.get("page", 1),
                chunk.page_content,
            )
        )

    if docs:
        vectorstore.add_documents(documents=docs, ids=ids)

    return len(chunks)


def list_documents_in_folder(data_dir: str | Path) -> list[Path]:
    """Return all supported files from the configured data folder."""
    folder = Path(data_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Documents folder does not exist: {folder}")
    files = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files


def ingest_documents(data_dir: str | Path, persist_dir: str | Path, chunk_size: int = 500, overlap: int = 50) -> dict:
    """Ingest all supported files in a folder and maintain a simple skip manifest."""
    folder = Path(data_dir)
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    manifest_path = persist_path / "ingest_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    ingested = []
    skipped = []

    for file_path in list_documents_in_folder(folder):
        source = _normalize_source_path(file_path)
        current_hash = _content_hash(file_path)
        if manifest.get(source) == current_hash:
            skipped.append(source)
            continue

        documents = load_document(str(file_path))
        chunks = split_documents(documents, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue
        embed_and_store(chunks, str(persist_path))
        manifest[source] = current_hash
        ingested.append(source)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"ingested": ingested, "skipped": skipped}


def ingest_document(filepath: str, persist_dir: str, chunk_size: int = 500, overlap: int = 50) -> int:
    """Ingest a single document into the vector store."""
    print(f"Loading document from {filepath}...")
    documents = load_document(filepath)
    print(f"Loaded {len(documents)} document(s).")

    print("Splitting documents into chunks...")
    chunks = split_documents(documents, chunk_size=chunk_size, overlap=overlap)
    print(f"Created {len(chunks)} chunks.")

    print(f"Embedding and storing in {persist_dir}...")
    num_chunks = embed_and_store(chunks, persist_dir)
    print(f"\nSuccessfully embedded {num_chunks} chunks!")
    return num_chunks


if __name__ == "__main__":
    if not config.DATA_DIR.exists():
        raise FileNotFoundError(f"Documents folder does not exist: {config.DATA_DIR}")

    result = ingest_documents(
        data_dir=config.DATA_DIR,
        persist_dir=config.CHROMA_PERSIST_DIR,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.OVERLAP,
    )
    print(f"Ingested {len(result['ingested'])} file(s); skipped {len(result['skipped'])} unchanged file(s).")
