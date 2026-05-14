from langchain_community.tools.tavily_search import (
    TavilySearchResults
)

from langchain_core.documents import Document

from workflow.state import GraphState


web_search_tool = TavilySearchResults(
    k=3
)


def web_search(state: GraphState):

    question = state["question"]

    docs = web_search_tool.invoke({
        "query": question
    })

    web_results = "\n".join(
        [doc["content"] for doc in docs]
    )

    web_document = Document(
        page_content=web_results
    )

    return {
        "question": question,
        "documents": [web_document]
    }