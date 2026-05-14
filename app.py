import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
    "Upload Documents",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

query = st.text_input(
    "Ask a question about your uploaded documents"
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

    for file in uploaded_files:
        st.write(f"📄 {file.name}")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        st.write(
            "RAG pipeline integration coming in next phase..."
        )