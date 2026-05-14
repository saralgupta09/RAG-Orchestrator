import os
import time
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

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Adaptive RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .stChatMessage {
            padding: 12px;
            border-radius: 12px;
        }

        .st-emotion-cache-1c7y2kd {
            border-radius: 12px;
        }

        .block-container {
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("⚙️ Settings")

    st.markdown("---")

    st.subheader("📄 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):

        st.session_state.messages = []

        st.success("Chat history cleared")

# ==========================================
# MAIN HEADER
# ==========================================

st.title("🤖 Adaptive RAG Assistant")

st.markdown(
    """
    An advanced Retrieval-Augmented Generation (RAG)
    system powered by LangGraph, LangChain,
    ChromaDB, and NVIDIA-hosted LLMs.
    """
)

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# ==========================================
# DOCUMENT INGESTION
# ==========================================

all_chunks = []

if uploaded_files:

    with st.spinner("Processing documents..."):

        for uploaded_file in uploaded_files:

            save_path = os.path.join(
                UPLOAD_DIR,
                uploaded_file.name
            )

            with open(save_path, "wb") as f:

                f.write(uploaded_file.getbuffer())

            documents = load_pdf_documents(
                save_path
            )

            if not documents:

                st.warning(
                    f"No readable text found in "
                    f"{uploaded_file.name}"
                )

                continue

            chunks = split_documents(
                documents
            )

            all_chunks.extend(chunks)

        # Create vector store
        if all_chunks:

            create_vector_store(all_chunks)

            st.success(
                f"Processed {len(uploaded_files)} "
                f"document(s) successfully"
            )

# ==========================================
# CHAT SECTION
# ==========================================

st.markdown("---")

st.subheader("💬 Chat with Your Documents")

# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# User input
query = st.chat_input(
    "Ask questions about your uploaded documents..."
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

        start_time = time.time()

        rag_graph = build_rag_graph()

        with st.spinner(
            "Running adaptive RAG workflow..."
        ):

            result = rag_graph.invoke({
                "question": query
            })

        response = result["generation"]

        sources = result.get(
            "sources",
            []
        )

        retrieved_docs = result["documents"]

        end_time = time.time()

        response_time = round(
            end_time - start_time,
            2
        )

        # Store assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        # Display assistant response
        with st.chat_message("assistant"):

            st.markdown(response)

            st.caption(
                f"Response generated in "
                f"{response_time} seconds"
            )

            # Sources
            with st.expander(
                "📚 Retrieved Sources"
            ):

                for source in sources:

                    st.markdown(
                        f"### Source "
                        f"{source['source_id']}"
                    )

                    st.write(
                        source["content"]
                    )

                    st.divider()

    except Exception as e:

        st.error(
            f"Workflow error: {str(e)}"
        )