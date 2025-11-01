from ddgs import DDGS
from langchain.tools import ToolRuntime, tool

from schemas.tool_ddgs import DDGSSearchInput, WebSearchSchemas


@tool("websearch", args_schema=DDGSSearchInput)
def ddgs_search(query: str, page: int = 1, runtime: ToolRuntime | None = None) -> list[WebSearchSchemas]:
    """Perform a web search for the user's question.
    You need to understand the user's intent and find the information and answer they're looking for.

    Args:
        query(str): A query to search the web for the user's question
        page(int): Pages when searching the web, default 1 (for each page, max_result 10)
    """
    if runtime:
        writer = runtime.stream_writer
        writer("🌐 Start Web Search")
    results = DDGS().text(
        query=query,
        region="kr-kr",
        max_results=10,
        page=page,
        backend="auto",
    )

    if results and runtime:
        writer(f"🌐 Finish Web Search: {len(results)} 문서 찾음, Page: {page}")
    return [WebSearchSchemas(**data) for data in results]
