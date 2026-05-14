from langgraph.graph import StateGraph
from langgraph.graph import END

from workflow.state import GraphState

from workflow.retrieve_node import retrieve
from workflow.grade_documents_node import (
    grade_documents
)
from workflow.generate_node import generate


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
        END
    )

    app = workflow.compile()

    return app