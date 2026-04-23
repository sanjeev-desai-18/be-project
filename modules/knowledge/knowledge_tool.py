# modules/knowledge/knowledge_tool.py
# ─────────────────────────────────────────────────────────────────────────────
# Ported from final-year-project reference into be-project-main.
# Changes vs original be-project knowledge_tool.py:
#   1. Caps combined web result at 1500 characters to avoid bloating LLM prompts.
#   2. Simpler, cleaner error handling — single try/catch, no multi-backend loop.
#      (The multi-backend loop in the original was added to handle rate limits;
#       the newer duckduckgo_search library handles backend selection internally.)
# ─────────────────────────────────────────────────────────────────────────────

from utils.logger import logger


def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web using DDGS.
    Falls back gracefully — never crashes the pipeline.
    Result is capped at 1500 chars to avoid bloating the LLM prompt.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if results:
            snippets = [r.get("body", "") for r in results if r.get("body")]
            if snippets:
                combined = " ".join(snippets)
                logger.info(f"Web search OK — {len(snippets)} results")
                return combined[:1500]   # cap to avoid huge prompts
    except Exception as e:
        logger.warning(f"Web search failed: {e}")

    return ""
