import os
import streamlit as st
from dotenv import load_dotenv

from workflow.rag_graph import build_rag_graph

from ingestion.document_loader import load_pdf_documents
from ingestion.text_splitter import split_documents

from vectorstore.chroma_store import (
    create_vector_store
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

# =========================
# FILE UPLOAD
# =========================

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
                f"No readable text found in "
                f"{uploaded_file.name}"
            )

            continue

        chunks = split_documents(documents)

        st.write(f"Chunks created: {len(chunks)}")

        all_chunks.extend(chunks)

        st.success(
            f"{uploaded_file.name} processed successfully"
        )

    # Create vector database
    if all_chunks:

        with st.spinner("Creating vector embeddings..."):

            create_vector_store(all_chunks)

        st.success("Vector store created successfully")

    st.write(
        f"Total chunks created: {len(all_chunks)}"
    )

    # Chunk preview
    with st.expander("Preview Chunks"):

        for chunk in all_chunks[:3]:

            st.write(
                chunk.page_content[:500]
            )

            st.divider()

# =========================
# LANGGRAPH CHAT INTERFACE
# =========================

st.divider()

st.subheader("Chat with your documents")

# Session state for chat history
if "messages" not in st.session_state:

    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# User input
query = st.chat_input(
    "Ask a question about your uploaded documents"
)

if query:

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Display user message
    with st.chat_message("user"):

        st.markdown(query)

    try:

        # Build graph
        rag_graph = build_rag_graph()

        with st.spinner("Running LangGraph workflow..."):

            result = rag_graph.invoke({
                "question": query
            })

        response = result["generation"]

        retrieved_docs = result["documents"]

        # Store assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        # Display assistant response
        with st.chat_message("assistant"):

            st.markdown(response)

            # Retrieved context
            with st.expander("Retrieved Context"):

                for i, doc in enumerate(retrieved_docs):

                    st.markdown(
                        f"### Result {i + 1}"
                    )

                    st.write(
                        doc.page_content[:1000]
                    )

                    st.divider()

    except Exception as e:

        st.error(
            f"Workflow error: {str(e)}"
        )