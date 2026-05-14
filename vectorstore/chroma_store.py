from langchain_chroma import Chroma

from vectorstore.embedding_model import (
    get_embedding_model
)


CHROMA_DIR = ".chromadb"


def create_vector_store(documents):

    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )

    return vector_store


def load_vector_store():

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )

    return vector_store