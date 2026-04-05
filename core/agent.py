import json
import re
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from core.state import AssistantState
from core.confidence import (
    get_confidence_zone,
    build_clarification_question,
    build_medium_prefix
)
from config import AGENT_MODEL, AGENT_TEMPERATURE, GROQ_API_KEY
from utils.logger import logger


# ── Groq LLM ─────────────────────────────────────────
llm = ChatGroq(
    model=AGENT_MODEL,
    temperature=AGENT_TEMPERATURE,
    api_key=GROQ_API_KEY
)


# ── Routing Prompt ────────────────────────────────────
ROUTING_PROMPT = """
You are the routing brain of a voice assistant for visually impaired users in India.

User said: "{transcript}"

Pick ONE mode from this list:
- navigation_mode
- reading_mode
- currency_mode
- stop_mode
- knowledge_mode
- greeting_mode
- unknown

Hints:
- paisa, note, money, currency, kitne ka → currency_mode
- stop, band karo, ruk jao, bas → stop_mode
- read, padho, kya likha hai → reading_mode
- surroundings, aas paas, bata kya hai, describe, surrounding → navigation_mode
- news, weather, time, who is, what is, information, update → knowledge_mode
- thank you, thanks, shukriya, dhanyawad, great, awesome, helpful → greeting_mode

IMPORTANT: Return ONLY a raw JSON object. No markdown. No code fences. No explanation.
Example: {{"mode": "navigation_mode", "confidence": 0.92, "cleaned_text": "describe surroundings", "extra_context": ""}}
"""

VALID_MODES = {
    "navigation_mode",
    "reading_mode",
    "currency_mode",
    "stop_mode",
    "knowledge_mode",
    "greeting_mode",
    "unknown"
}


# ── Safe JSON extractor ───────────────────────────────
def _parse_llm_json(raw: str) -> dict:
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {raw!r}")
    result = json.loads(match.group())
    return {k.strip(): v for k, v in result.items()}


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA OWNERSHIP HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _stop_currency_if_running() -> bool:
    """
    Stop currency detection if active and wait for camera to be fully released.
    Returns True if currency was stopped, False if it wasn't running.
    """
    try:
        import modules.currency.currency_module as _cm
        import modules.currency.currency_detector as _det

        if not _cm.currency_active:
            return False

        logger.info("Camera requested by another module — stopping currency first...")
        _cm.stop_currency_mode()

        released = _det._camera_released.wait(timeout=2.0)
        if not released:
            logger.error("Camera still not released after stop — proceeding anyway")

        logger.info("Currency stopped, camera free ✓")
        return True

    except Exception as e:
        logger.error(f"_stop_currency_if_running failed: {e}", exc_info=True)
        return False


# ═══════════════════════════════════════════════
# NODE 1 — Interpret Intent
# ═══════════════════════════════════════════════
def interpret_intent_node(state: AssistantState) -> AssistantState:
    transcript = state["raw_transcript"]

    fallback = {
        **state,
        "mode":                "unknown",
        "confidence":          0.0,
        "cleaned_transcript":  transcript,
        "extra_context":       "",
        "needs_clarification": False,
        "final_output":        "",
    }

    if not transcript.strip():
        logger.warning("Empty transcript — skipping LLM call")
        return fallback

    prompt = ROUTING_PROMPT.format(transcript=transcript)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        logger.debug(f"LLM raw output: {raw!r}")

        result = _parse_llm_json(raw)

        mode       = str(result.get("mode", "unknown")).strip()
        confidence = float(result.get("confidence", 0.0))
        cleaned    = str(result.get("cleaned_text", transcript)).strip()
        extra      = str(result.get("extra_context", "")).strip()

        if mode not in VALID_MODES:
            logger.warning(f"Unexpected mode '{mode}' — falling back to unknown")
            mode = "unknown"

        logger.info(f"Agent → mode: {mode} | confidence: {confidence:.2f}")

        return {
            **state,
            "mode":                mode,
            "confidence":          confidence,
            "cleaned_transcript":  cleaned,
            "extra_context":       extra,
            "needs_clarification": False,
            "final_output":        "",
        }

    except Exception as e:
        logger.error(f"Error in interpret_intent: {e}", exc_info=True)
        return fallback


# ═══════════════════════════════════════════════
# NODE 2 — Confidence Router
# ═══════════════════════════════════════════════
def confidence_router_node(state: AssistantState) -> AssistantState:
    zone = get_confidence_zone(state["confidence"])

    if zone == "low" or state["mode"] == "unknown":
        question = build_clarification_question(state["mode"])
        return {**state, "needs_clarification": True, "final_output": question}
    elif zone == "medium":
        prefix = build_medium_prefix(state["mode"])
        return {**state, "final_output": prefix}
    else:
        return {**state, "final_output": ""}


# ═══════════════════════════════════════════════
# CONDITIONAL EDGE
# ═══════════════════════════════════════════════
def route_to_module(state: AssistantState) -> str:
    if state.get("needs_clarification"):
        return "tts_node"

    routes = {
        "navigation_mode": "scene_node",
        "reading_mode":    "reading_node",
        "currency_mode":   "currency_node",
        "stop_mode":       "stop_node",
        "knowledge_mode":  "knowledge_node",
        "greeting_mode":   "greeting_node",
    }
    return routes.get(state.get("mode", "unknown"), "tts_node")


# ═══════════════════════════════════════════════
# NODE 3a — Scene
# ═══════════════════════════════════════════════
def scene_node(state: AssistantState) -> AssistantState:
    from modules.scene.scene_module import SceneModule
    from tts.speaker import Speaker
    logger.info("Executing Scene module")

    _stop_currency_if_running()

    sp = Speaker()

    # Immediate ACK — user hears this within ~200ms of routing completing,
    # while the camera is still warming up and the VLM hasn't been called yet.
    sp.speak("Looking at your surroundings.")

    try:
        # speaker=sp causes each sentence to be spoken as it streams.
        result = SceneModule().run(speaker=sp)
    except Exception as e:
        logger.error(f"Scene module error: {e}", exc_info=True)
        result = "I was unable to analyse the scene."
        sp.speak_stream(result)

    # spoken=True tells tts_node to skip — audio already played sentence by sentence.
    return {**state, "final_output": result, "spoken": True}


# ═══════════════════════════════════════════════
# NODE 3b — Reading
# ═══════════════════════════════════════════════
def reading_node(state: AssistantState) -> AssistantState:
    from modules.reading.reading_module import ReadingModule
    from tts.speaker import Speaker
    logger.info("Executing Reading module")

    _stop_currency_if_running()

    sp = Speaker()

    # Immediate ACK
    sp.speak("Reading now.")

    try:
        # speaker=sp causes each sentence to be spoken as it streams.
        result = ReadingModule().run(speaker=sp)
    except Exception as e:
        logger.error(f"Reading module error: {e}", exc_info=True)
        result = "I could not read the text."
        sp.speak_stream(result)

    # spoken=True — tts_node will skip re-speaking.
    return {**state, "final_output": result, "spoken": True}


# ═══════════════════════════════════════════════
# NODE 3c — Currency
# ═══════════════════════════════════════════════
def currency_node(state: AssistantState) -> AssistantState:
    from modules.currency.currency_module import start_currency_mode, currency_active
    logger.info("Starting Currency continuous mode")

    try:
        if currency_active:
            logger.info("Currency already active — skipping duplicate start")
            return {**state, "final_output": ""}

        start_currency_mode()
        result = "Currency mode on."

    except Exception as e:
        logger.error(f"Currency module error: {e}", exc_info=True)
        result = "I could not start currency detection."

    return {**state, "final_output": result}


# ═══════════════════════════════════════════════
# NODE 3d — Stop
# ═══════════════════════════════════════════════
def stop_node(state: AssistantState) -> AssistantState:
    from modules.currency.currency_module import stop_currency_mode, currency_active
    logger.info("Stopping active modules")

    try:
        if not currency_active:
            logger.info("Nothing active to stop")
            return {**state, "final_output": ""}

        stop_currency_mode()
        result = "Stopped."

    except Exception as e:
        logger.error(f"Stop error: {e}", exc_info=True)
        result = "Could not stop."

    return {**state, "final_output": result}


# ═══════════════════════════════════════════════
# NODE 3e — Knowledge
# ═══════════════════════════════════════════════
def knowledge_node(state: AssistantState) -> AssistantState:
    from modules.knowledge.knowledge_logic import handle_knowledge_query
    logger.info("Executing Knowledge module")

    try:
        query = state.get("cleaned_transcript", "")
        handle_knowledge_query(query)
        return {**state, "final_output": ""}

    except Exception as e:
        logger.error(f"Knowledge module error: {e}", exc_info=True)
        return {**state, "final_output": "", "spoken": True}


# ═══════════════════════════════════════════════
# NODE 3f — Greeting
# ═══════════════════════════════════════════════
def greeting_node(state: AssistantState) -> AssistantState:
    logger.info("Executing Greeting node")
    return {
        **state,
        "final_output": "I hope everything is alright! How can I help you more?"
    }


# ═══════════════════════════════════════════════
# NODE 4 — TTS
# ═══════════════════════════════════════════════
def tts_node(state: AssistantState) -> AssistantState:
    from tts.speaker import Speaker

    # Skip if the module already spoke (scene_node / reading_node stream directly)
    if state.get("spoken"):
        logger.debug("tts_node: spoken=True — skipping (audio already played)")
        return state

    output = state.get("final_output", "").strip()
    if output:
        Speaker().speak(output)
    return state


# ═══════════════════════════════════════════════
# BUILD GRAPH
# ═══════════════════════════════════════════════
def build_agent():
    graph = StateGraph(AssistantState)

    graph.add_node("interpret_intent",  interpret_intent_node)
    graph.add_node("confidence_router", confidence_router_node)
    graph.add_node("scene_node",        scene_node)
    graph.add_node("reading_node",      reading_node)
    graph.add_node("currency_node",     currency_node)
    graph.add_node("stop_node",         stop_node)
    graph.add_node("knowledge_node",    knowledge_node)
    graph.add_node("greeting_node",     greeting_node)
    graph.add_node("tts_node",          tts_node)

    graph.set_entry_point("interpret_intent")
    graph.add_edge("interpret_intent", "confidence_router")

    graph.add_conditional_edges(
        "confidence_router",
        route_to_module,
        {
            "scene_node":     "scene_node",
            "reading_node":   "reading_node",
            "currency_node":  "currency_node",
            "stop_node":      "stop_node",
            "knowledge_node": "knowledge_node",
            "greeting_node":  "greeting_node",
            "tts_node":       "tts_node",
        }
    )

    graph.add_edge("scene_node",     "tts_node")
    graph.add_edge("reading_node",   "tts_node")
    graph.add_edge("currency_node",  "tts_node")
    graph.add_edge("stop_node",      "tts_node")
    graph.add_edge("knowledge_node", "tts_node")
    graph.add_edge("greeting_node",  "tts_node")
    graph.add_edge("tts_node", END)

    return graph.compile()


agent = build_agent()
logger.info("LangGraph agent compiled and ready ✓")
