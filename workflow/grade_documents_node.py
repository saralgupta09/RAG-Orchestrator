from workflow.state import GraphState

from chains.document_grader import (
    document_grader
)


def grade_documents(state: GraphState):

    question = state["question"]

    documents = state["documents"]

    filtered_docs = []

    for doc in documents:

        score = document_grader.invoke({
            "question": question,
            "document": doc.page_content
        })

        if score.binary_score == "yes":

            filtered_docs.append(doc)

    return {
        "question": question,
        "documents": filtered_docs
    }