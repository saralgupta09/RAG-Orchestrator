from workflow.state import GraphState

from vectorstore.chroma_store import (
    load_vector_store
)


def retrieve(state: GraphState):

    question = state["question"]

    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    documents = retriever.invoke(question)

    return {
        "question": question,
        "documents": documents
    }