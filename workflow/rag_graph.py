from langgraph.graph import StateGraph
from langgraph.graph import END

from workflow.state import GraphState

from workflow.retrieve_node import retrieve
from workflow.grade_documents_node import (
    grade_documents
)
from workflow.generate_node import generate
from workflow.hallucination_check_node import (
    hallucination_check
)


def build_rag_graph():

    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node(
        "retrieve",
        retrieve
    )

    workflow.add_node(
        "grade_documents",
        grade_documents
    )

    workflow.add_node(
        "generate",
        generate
    )

    workflow.add_node(
        "hallucination_check",
        hallucination_check
    )

    # Entry point
    workflow.set_entry_point("retrieve")

    # Flow
    workflow.add_edge(
        "retrieve",
        "grade_documents"
    )

    workflow.add_edge(
        "grade_documents",
        "generate"
    )

    workflow.add_edge(
        "generate",
        "hallucination_check"
    )

    workflow.add_edge(
        "hallucination_check",
        END
    )

    app = workflow.compile()

    return app