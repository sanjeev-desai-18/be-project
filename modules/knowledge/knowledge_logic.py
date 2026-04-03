# modules/knowledge/knowledge_logic.py

from utils.logger import logger
from tts.speaker import Speaker
from modules.knowledge.knowledge_tool import search_web
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import AGENT_MODEL, GROQ_API_KEY
from langdetect import detect
import datetime
import requests

llm = ChatGroq(
    model=AGENT_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.2
)

# ─────────────────────────────────────────────
# Keywords — skip web search for these
# ─────────────────────────────────────────────
TIME_KEYWORDS    = ["time", "समय", "वेळ", "बजे"]
DATE_KEYWORDS    = ["date", "day", "today", "तारीख", "दिनांक", "आज"]
WEATHER_KEYWORDS = ["weather", "temperature", "temp", "मौसम", "हवामान"]

def _needs_web_search(query: str) -> bool:
    q = query.lower()
    local_keywords = TIME_KEYWORDS + DATE_KEYWORDS + WEATHER_KEYWORDS
    return not any(k in q for k in local_keywords)

def _needs_weather(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in WEATHER_KEYWORDS)


# ─────────────────────────────────────────────
# Language Detection
# ─────────────────────────────────────────────
def _detect_language(query: str) -> str:
    try:
        code = detect(query)
        logger.info(f"Detected language code: {code}")
        return code
    except Exception:
        return "en"


# ─────────────────────────────────────────────
# Local Data Fetchers
# ─────────────────────────────────────────────
def _get_current_time() -> str:
    return datetime.datetime.now().strftime("%I:%M %p")

def _get_current_date() -> str:
    return datetime.datetime.now().strftime("%A, %d %B %Y")

def _get_weather() -> str:
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=19.2183&longitude=73.0868&current_weather=true"
        data = requests.get(url, timeout=5).json()
        temp = data["current_weather"]["temperature"]
        return f"{temp} degrees Celsius"
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return "unavailable"


# ─────────────────────────────────────────────
# LLM Call
# ─────────────────────────────────────────────
def _ask_llm(query: str, lang_code: str, web_context: str = "", weather: str = "") -> str:
    now_time   = _get_current_time()
    today_date = _get_current_date()

    context_section = (
        f"Additional web context: {web_context}"
        if web_context else
        "No web context available — use your own knowledge."
    )

    weather_section = (
        f"- Current temperature (Dombivli): {weather}"
        if weather else ""
    )

    prompt = f"""You are a helpful voice assistant for visually impaired users in India.

LANGUAGE RULE: Reply in language with ISO code '{lang_code}' only.

LIVE DATA:
- Current time: {now_time}
- Today's date: {today_date}
{weather_section}

{context_section}

Rules:
- Answer in MAXIMUM 2 short, natural sentences.
- Give the direct answer immediately — no preamble.
- Speak naturally as if talking to a person out loud.

Question: {query}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().replace("\n", " ")


# ─────────────────────────────────────────────
# Main Handler
# ─────────────────────────────────────────────
def handle_knowledge_query(query: str):
    try:
        logger.info(f"Knowledge query: {query}")

        lang_code = _detect_language(query)
        weather   = _get_weather() if _needs_weather(query) else ""

        web_context = ""
        if _needs_web_search(query):
            try:
                web_context = search_web(query)
                if web_context:
                    logger.info("Web search succeeded — using as context boost")
                else:
                    logger.warning("Web search empty — LLM will use own knowledge")
            except Exception as e:
                logger.warning(f"Web search skipped: {e}")

        answer = _ask_llm(query, lang_code, web_context, weather)
        Speaker().speak(answer)

    except Exception as e:
        logger.error(f"Knowledge logic error: {e}", exc_info=True)
        try:
            Speaker().speak("Sorry, I couldn't process that.")
        except Exception:
            pass
