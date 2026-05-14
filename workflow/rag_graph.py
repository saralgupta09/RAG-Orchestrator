from langgraph.graph import StateGraph
from langgraph.graph import END

from workflow.state import GraphState

from workflow.retrieve_node import retrieve
from workflow.generate_node import generate


def build_rag_graph():

    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node(
        "retrieve",
        retrieve
    )

    workflow.add_node(
        "generate",
        generate
    )

    # Flow
    workflow.set_entry_point("retrieve")

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    app = workflow.compile()

    return app