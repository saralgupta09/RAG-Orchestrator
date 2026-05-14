from workflow.state import GraphState

from chains.hallucination_grader import (
    hallucination_grader
)


def hallucination_check(state: GraphState):

    documents = state["documents"]

    generation = state["generation"]

    score = hallucination_grader.invoke({
        "documents": "\n\n".join(
            [doc.page_content for doc in documents]
        ),
        "generation": generation
    })

    if score.binary_score == "yes":

        return {
            "question": state["question"],
            "documents": documents,
            "generation": generation
        }

    return {
        "question": state["question"],
        "documents": documents,
        "generation": (
            "The generated answer could not be "
            "verified from the retrieved documents."
        )
    }