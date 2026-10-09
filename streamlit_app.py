from pathlib import Path

import streamlit as st

try:
    from src import config
    from src.chatbot import chat
    from src.ingest import ingest_documents
except ModuleNotFoundError:
    import config
    from chatbot import chat
    from ingest import ingest_documents


st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader(
    "Upload documents to ingest",
    accept_multiple_files=True,
    type=["txt", "md", "pdf"],
)

if uploaded_files:
    temp_dir = Path(config.DATA_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        target = temp_dir / safe_name
        target.write_bytes(uploaded_file.read())

    ingest_documents(temp_dir, config.CHROMA_PERSIST_DIR, chunk_size=config.CHUNK_SIZE, overlap=config.OVERLAP)
    st.success(f"Ingested {len(uploaded_files)} file(s).")

query = st.chat_input("Ask a question about your documents")

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.caption("Sources: " + ", ".join(message["sources"]))

if query:
    result = chat(
        query,
        config.CHROMA_PERSIST_DIR,
        top_k=config.TOP_K,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
        chat_history=st.session_state.history,
    )

    user_message = {"role": "user", "content": query, "sources": []}
    assistant_message = {
        "role": "assistant",
        "content": result["response"],
        "sources": result["sources"],
    }

    st.session_state.history.append(user_message)
    st.session_state.history.append(assistant_message)

    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        st.markdown(result["response"])
        if result["sources"]:
            st.caption("Sources: " + ", ".join(result["sources"]))
