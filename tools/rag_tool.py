
from langchain_core.tools import tool

from rag.retriever import retrieve


@tool
def rag_search(query: str) -> str:
    """
    Search the internal knowledge base for relevant
    documentation, support policies, project information,
    and troubleshooting instructions.

    Use this tool when a user asks about information
    contained in the internal knowledge base.
    """
    results = retrieve(query, top_k=3)

    if not results:
        return "No relevant documents found in the knowledge base."

    formatted_results = []

    for index, result in enumerate(results, start=1):
        source = result.get("source") or "Unknown"
        text = result.get("text", "")

        formatted_results.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Content:\n{text}"
        )

    return "\n\n---\n\n".join(formatted_results)
