import os
import streamlit as st
from dotenv import load_dotenv
from chains.rag_chain import generate_answer

from ingestion.document_loader import load_pdf_documents
from ingestion.text_splitter import split_documents

from vectorstore.chroma_store import (
    create_vector_store,
    load_vector_store
)

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

all_chunks = []

# =========================
# DOCUMENT INGESTION
# =========================

if uploaded_files:

    for uploaded_file in uploaded_files:

        save_path = os.path.join(
            UPLOAD_DIR,
            uploaded_file.name
        )

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        documents = load_pdf_documents(save_path)

        st.write(f"Pages loaded: {len(documents)}")

        if not documents:
            st.warning(
                f"No readable text found in {uploaded_file.name}"
            )
            continue

        chunks = split_documents(documents)

        st.write(f"Chunks created: {len(chunks)}")

        all_chunks.extend(chunks)

        st.success(
            f"{uploaded_file.name} processed successfully"
        )

    # Create vector DB only once
    if all_chunks:

        with st.spinner("Creating vector embeddings..."):

            create_vector_store(all_chunks)

        st.success("Vector store created successfully")

    st.write(f"Total chunks created: {len(all_chunks)}")

    # Preview chunks
    with st.expander("Preview Chunks"):

        for chunk in all_chunks[:3]:

            st.write(chunk.page_content[:500])

            st.divider()

# =========================
# QUESTION ANSWERING
# =========================

st.divider()

st.subheader("Ask Questions")

query = st.text_input(
    "Enter your question"
)
if query:

    try:

        vector_store = load_vector_store()

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:

            st.warning("No relevant documents found.")

        else:

            with st.spinner("Generating answer..."):

                answer = generate_answer(
                    retrieved_docs,
                    query
                )

            st.subheader("Answer")

            st.write(answer)

            with st.expander("Retrieved Context"):

                for i, doc in enumerate(retrieved_docs):

                    st.markdown(f"### Result {i + 1}")

                    st.write(doc.page_content[:1000])

                    st.divider()

    except Exception as e:

        st.error(f"Retrieval error: {str(e)}")