from typing import TypedDict, Optional, Dict, Any


class AssistantState(TypedDict):
    """
    Single source of truth for one complete request cycle:
    Voice → STT → Agent → Module → TTS
    """

    # ── Input stage ───────────────────────────────────
    raw_transcript: str
    cleaned_transcript: str

    # ── Agent decision ────────────────────────────────
    mode: str
    confidence: float
    extra_context: Optional[str]

    # ── Module output ─────────────────────────────────
    raw_output: Optional[Dict[str, Any]]
    final_output: str

    # ── Control flags ─────────────────────────────────
    needs_clarification: bool
    clarification_question: Optional[str]
    error: Optional[str]
    retry_count: int

    # ── Runtime flags ─────────────────────────────────
    spoken: bool