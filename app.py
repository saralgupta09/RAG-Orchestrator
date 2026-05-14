import os
import streamlit as st
from dotenv import load_dotenv

from ingestion.document_loader import load_pdf_documents
from ingestion.text_splitter import split_documents

load_dotenv()

UPLOAD_DIR = "data/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="Adaptive RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Adaptive RAG Assistant")

st.markdown(
    """
    Upload documents and ask context-aware questions using
    LangGraph + LangChain + ChromaDB.
    """
)

uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []

    for uploaded_file in uploaded_files:

        save_path = os.path.join(
            UPLOAD_DIR,
            uploaded_file.name
        )

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        documents = load_pdf_documents(save_path)
        st.write(f"Pages loaded: {len(documents)}")

        chunks = split_documents(documents)
        st.write(f"Chunks created: {len(chunks)}")

        all_chunks.extend(chunks)

        st.success(
            f"{uploaded_file.name} processed successfully"
        )

    st.write(f"Total chunks created: {len(all_chunks)}")

    with st.expander("Preview Chunks"):

        for chunk in all_chunks[:3]:
            st.write(chunk.page_content[:500])
            st.divider()