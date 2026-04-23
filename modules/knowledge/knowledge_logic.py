# modules/knowledge/knowledge_logic.py
# ─────────────────────────────────────────────────────────────────────────────
# Ported from final-year-project reference into be-project-main.
# Changes vs original be-project knowledge_logic.py:
#   1. Uses dedicated KNOWLEDGE_MODEL (llama-3.3-70b-versatile) instead of
#      AGENT_MODEL — much stronger for factual Q&A.
#   2. Conversation history (last 3 exchanges = 6 messages) for follow-up support.
#   3. TIME / DATE queries answered instantly from system clock (no LLM call).
#   4. Weather response now includes a weather description (not just temperature).
#   5. LLM prompt uses SystemMessage + history + HumanMessage (multi-turn context).
#   6. Parallel execution: language detection + weather + web search run together.
#   7. Web search result capped at 1500 chars to avoid huge prompts.
#   8. Empty / punctuation-only query guard at the top.
#   9. Broader multilingual keyword lists.
#  10. handle_knowledge_query() now returns str — fully backward-compatible since
#      BE's agent.py knowledge_node ignores the return value anyway.
# ─────────────────────────────────────────────────────────────────────────────

from utils.logger import logger
from tts.speaker import Speaker
from modules.knowledge.knowledge_tool import search_web
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import GROQ_API_KEY
from langdetect import detect
import datetime
import requests
import concurrent.futures

# ── Dedicated model for knowledge (stronger than the routing agent model) ──────
KNOWLEDGE_MODEL = "llama-3.3-70b-versatile"

llm = ChatGroq(model=KNOWLEDGE_MODEL, api_key=GROQ_API_KEY, temperature=0.3)

# ── Conversation history (last 3 exchanges = 6 messages) ─────────────────────
_history: list = []

def _add_to_history(role: str, content: str):
    _history.append({"role": role, "content": content})
    if len(_history) > 6:
        _history.pop(0)


# ── Keyword lists (broader than original — includes Hindi/Marathi/Roman Hindi) ─
_TIME_KW    = ["time", "समय", "वेळ", "बजे", "kitne baje", "what time"]
_DATE_KW    = ["date", "day", "today", "तारीख", "दिनांक", "आज", "what day"]
_WEATHER_KW = [
    "weather", "temperature", "temp", "मौसम", "हवामान",
    "garmi", "thand", "rain", "hot", "cold"
]

def _is_local_query(q: str) -> bool:
    """Time/date queries — answer from system clock, no LLM or web needed."""
    return any(k in q for k in _TIME_KW + _DATE_KW)

def _needs_weather(q: str) -> bool:
    return any(k in q for k in _WEATHER_KW)

def _needs_web(q: str) -> bool:
    """Always search web EXCEPT for pure time/date questions."""
    return not _is_local_query(q)


# ── Local data fetchers ────────────────────────────────────────────────────────
def _get_time() -> str:
    return datetime.datetime.now().strftime("%I:%M %p")

def _get_date() -> str:
    return datetime.datetime.now().strftime("%A, %d %B %Y")

def _get_weather() -> str:
    """Returns temperature + human-readable weather description for Dombivli."""
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=19.2183&longitude=73.0868&current_weather=true"
        )
        data = requests.get(url, timeout=4).json()
        temp = data["current_weather"]["temperature"]
        code = data["current_weather"]["weathercode"]
        desc = {
            0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
            45: "foggy", 51: "light drizzle", 61: "light rain", 63: "moderate rain",
            65: "heavy rain", 80: "rain showers", 95: "thunderstorm"
        }.get(code, "")
        return f"{temp}°C{', ' + desc if desc else ''}"
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return ""

def _detect_lang(query: str) -> str:
    try:
        return detect(query)
    except Exception:
        return "en"


# ── LLM call with multi-turn history ──────────────────────────────────────────
def _ask_llm(query: str, lang: str, web: str = "", weather: str = "") -> str:
    now_time   = _get_time()
    today_date = _get_date()

    live = f"- Time: {now_time}\n- Date: {today_date}"
    if weather:
        live += f"\n- Weather (Dombivli): {weather}"

    web_section = f"\nLatest web results:\n{web}" if web else ""

    system = f"""You are a helpful voice assistant for a visually impaired person in India.

LIVE DATA:
{live}
{web_section}

RULES:
- Reply ONLY in language code '{lang}'.
- Maximum 2 short sentences. This is spoken audio — keep it brief.
- Answer directly. No preamble like "Sure", "According to", "Of course".
- Use web results if provided — they have the latest info.
- If web results are empty, use your own knowledge confidently.
- Never say "I don't know" or "not specified" — give your best answer.
- The user speaks via microphone so transcription errors are common. Use \
conversation history and context to infer the correct meaning \
(e.g. "Chember" = Chembur, "Wased" = VESIT). Always interpret charitably."""

    messages = [SystemMessage(content=system)]
    for m in _history:
        messages.append(
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
        )
    messages.append(HumanMessage(content=query))

    return llm.invoke(messages).content.strip().replace("\n", " ")


# ── Main handler ───────────────────────────────────────────────────────────────
def handle_knowledge_query(query: str) -> str:
    """
    Handle a knowledge query end-to-end: fetch data, call LLM, speak the answer.
    Returns the answer string (BE's agent.py ignores the return value, but it's
    available for future use or testing).
    """
    try:
        query = query.strip()
        if not query or query in [".", ",", "?", "!", ""]:
            logger.warning("Empty knowledge query — skipping")
            return ""

        logger.info(f"Knowledge: '{query}'")
        q = query.lower()

        # ── Instant time/date replies (no LLM needed) ────────────────────────
        if any(k in q for k in _TIME_KW) and not _needs_weather(q):
            answer = f"It is {_get_time()}."
            Speaker().speak(answer)
            _add_to_history("user", query)
            _add_to_history("assistant", answer)
            return answer

        if any(k in q for k in _DATE_KW) and not _needs_weather(q):
            answer = f"Today is {_get_date()}."
            Speaker().speak(answer)
            _add_to_history("user", query)
            _add_to_history("assistant", answer)
            return answer

        # ── Parallel: language detection + weather + web search ───────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            lang_f    = ex.submit(_detect_lang, query)
            weather_f = ex.submit(_get_weather) if _needs_weather(q) else None
            web_f     = ex.submit(search_web, query) if _needs_web(q) else None

            lang    = lang_f.result()
            weather = weather_f.result(timeout=5) if weather_f else ""
            web     = ""
            if web_f:
                try:
                    web = web_f.result(timeout=7)
                except Exception as e:
                    logger.warning(f"Web search timed out: {e}")

        answer = _ask_llm(query, lang, web, weather)

        _add_to_history("user", query)
        _add_to_history("assistant", answer)

        logger.info(f"Answer: '{answer[:80]}'")
        Speaker().speak(answer)
        return answer

    except Exception as e:
        logger.error(f"Knowledge error: {e}", exc_info=True)
        Speaker().speak("Sorry, I couldn't get that.")
        return "Sorry, I couldn't get that."
