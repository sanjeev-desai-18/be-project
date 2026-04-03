# modules/knowledge/knowledge_tool.py

from duckduckgo_search import DDGS
from utils.logger import logger


def search_web(query: str, max_results: int = 3) -> str:
    """
    Try DuckDuckGo with multiple backends.
    Returns snippet string or empty string on failure.
    """
    backends = ["html", "lite"]  # skip "api" — it always rate limits

    for backend in backends:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results, backend=backend))

            if results:
                snippets = [r.get("body", "") for r in results if r.get("body")]
                if snippets:
                    logger.info(f"Search success via backend: {backend}")
                    return " ".join(snippets)

        except Exception as e:
            logger.warning(f"Search failed with backend '{backend}': {e}")
            continue

    logger.error("All search backends failed")
    return ""
