import os

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not configured.")


client = TavilyClient(api_key=TAVILY_API_KEY)


@tool
def web_search(query: str) -> str:
    """
    Search the internet for current or external information.

    Use this tool when the user's question requires:
    - current information
    - recent developments
    - information not available in the knowledge base
    - general web research
    """

    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
        )

        results = response.get("results", [])

        if not results:
            return "No relevant web search results were found."

        output = []

        for result in results:
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("content", "")

            output.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"Web search error: {e}"