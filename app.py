import os
import shutil

import chromadb
import streamlit as st

from langchain_chroma import Chroma

from config import embeddings, llm
from ingest import ingest_document

st.set_page_config(page_title="Simple RAG")

st.title("Simple RAG Demo")

MAX_TURNS = 5
CHROMA_DIR = "chroma_db"
UPLOAD_DIR = "uploaded_docs"

if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "document_ready" not in st.session_state:
    st.session_state.document_ready = os.path.exists(CHROMA_DIR) and bool(os.listdir(CHROMA_DIR))
if "current_doc_name" not in st.session_state:
    st.session_state.current_doc_name = None


def reset_chat():
    st.session_state.turn_count = 0


def extract_answer_text(content) -> str:
    """
    Normalizes an LLM response's .content into plain text.
    Some providers/versions return a plain string; others return a list
    of content blocks like [{"type": "text", "text": "..."}]. This handles
    both so the UI always shows readable text, not a raw dict/list.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)

    return str(content)


st.sidebar.header("Document")

uploaded_file = st.sidebar.file_uploader(
    "Add your document (one at a time)",
    type=["pdf"],  # adjust to whatever file types your ingest.py supports
    accept_multiple_files=False,
)

if uploaded_file is not None:
    if st.session_state.current_doc_name != uploaded_file.name:
        # A new document was selected -> clear out the old vectorstore/upload first
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        saved_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with st.spinner("Processing document..."):
                ingest_document(saved_path, persist_directory=CHROMA_DIR)
        except ValueError as e:
            st.sidebar.error(str(e))
            # Clean up so the failed file/db don't linger in a half-done state
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
            if os.path.exists(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
            st.session_state.current_doc_name = None
            st.session_state.document_ready = False
            st.stop()

        st.session_state.current_doc_name = uploaded_file.name
        st.session_state.document_ready = True
        reset_chat()
        st.sidebar.success(f"'{uploaded_file.name}' ingested successfully.")
    else:
        st.sidebar.info(f"Using previously ingested document: {uploaded_file.name}")

if st.session_state.current_doc_name:
    st.sidebar.caption(f"Current document: **{st.session_state.current_doc_name}**")
    if st.sidebar.button("Remove document"):
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        st.session_state.current_doc_name = None
        st.session_state.document_ready = False
        reset_chat()
        st.rerun()

if not st.session_state.document_ready:
    st.info("Please upload a document in the sidebar to get started.")
    st.stop()

remaining = MAX_TURNS - st.session_state.turn_count
st.caption(f"Questions remaining this session: {remaining}/{MAX_TURNS}")

if remaining <= 0:
    st.warning(
        f"You've reached the limit of {MAX_TURNS} questions for this session. "
        "Upload a new document or refresh the page to reset."
    )
    st.stop()

chromadb.api.client.SharedSystemClient.clear_system_cache()

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

question = st.text_input("Ask a question about the document")

if question:
    st.session_state.turn_count += 1

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    answer_text = extract_answer_text(response.content)

    st.subheader("Answer")

    st.markdown(answer_text)

    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"### Chunk {i}")
            st.write(doc.page_content)

    st.caption(f"Questions used: {st.session_state.turn_count}/{MAX_TURNS}")