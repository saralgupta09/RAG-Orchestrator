from workflow.state import GraphState

from chains.rag_chain import generate_answer


def generate(state: GraphState):

    question = state["question"]

    documents = state["documents"]

    generation = generate_answer(
        documents,
        question
    )

    sources = []

    for i, doc in enumerate(documents):

        sources.append({
            "source_id": i + 1,
            "content": doc.page_content[:500]
        })

    return {
        "question": question,
        "documents": documents,
        "generation": generation,
        "sources": sources
    }