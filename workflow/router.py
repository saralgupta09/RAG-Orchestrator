from workflow.state import GraphState


def route_question(state: GraphState):

    documents = state["documents"]

    if not documents:

        return "web_search"

    return "generate"