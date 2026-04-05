# modules/scene/vlm_client.py
#
# Vision client using Groq's vision model.
#
# Two methods:
#   describe()        — blocking, returns full string. Used as fallback.
#   describe_stream() — generator, yields sentences as they arrive from the API.
#                       Use this for lowest latency — caller speaks each sentence
#                       immediately instead of waiting for the full response.

import time
from groq import Groq
from utils.logger import logger
from config import GROQ_API_KEY, VLM_MODEL, VLM_MAX_TOKENS


# Punctuation that marks a safe sentence boundary to speak on
_SENTENCE_ENDS = frozenset({".", "!", "?", "\n"})
# Speak on commas/semicolons only after enough chars have accumulated,
# so we don't emit tiny fragments
_CLAUSE_ENDS   = frozenset({",", ";"})
_CLAUSE_MIN_LEN = 40   # minimum buffer length before splitting on a clause end


class VLMClient:
    """
    Sends image + prompt to Groq Vision model.
    Used by scene and reading modules.
    """

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model  = VLM_MODEL
        logger.debug(f"VLMClient ready — model: {self.model}")

    # ── Blocking (original) ───────────────────────────────────────────────────

    def describe(self, image_b64: str, prompt: str) -> str:
        """
        Blocking call — waits for complete response.
        Use describe_stream() instead for lower latency.
        """
        logger.debug(f"Calling Groq Vision blocking ({self.model})...")
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=VLM_MAX_TOKENS,
                timeout=30,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            latency = time.time() - start
            logger.debug(f"VLM blocking latency: {latency:.2f}s")
            result = response.choices[0].message.content if response.choices else ""
            return result.strip()
        except Exception as e:
            logger.error(f"Groq Vision blocking call failed: {e}")
            return "I was unable to analyse the image right now. Please try again."

    # ── Streaming ─────────────────────────────────────────────────────────────

    def describe_stream(self, image_b64: str, prompt: str):
        """
        Generator — yields complete sentences as tokens arrive from the VLM.

        The caller iterates and can speak each yielded sentence immediately,
        so the user hears the first sentence while the VLM is still generating
        the rest. This is the primary path for scene and reading modules.

        Example:
            for sentence in vlm.describe_stream(frame, prompt):
                speaker.speak(sentence)

        Yields:
            str — one spoken sentence / clause at a time, stripped of whitespace.
            On error, yields a single error message string and returns.
        """
        logger.debug(f"Calling Groq Vision streaming ({self.model})...")
        start = time.time()

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                max_tokens=VLM_MAX_TOKENS,
                stream=True,
                timeout=30,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
        except Exception as e:
            logger.error(f"Groq Vision stream init failed: {e}")
            yield "I was unable to analyse the image right now. Please try again."
            return

        buffer = ""
        first_token_time = None

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is None:
                continue

            if first_token_time is None:
                first_token_time = time.time()
                logger.debug(f"VLM first token in {first_token_time - start:.2f}s")

            buffer += delta

            # Drain all complete sentences from the front of the buffer
            while True:
                idx = -1
                for i, ch in enumerate(buffer):
                    if ch in _SENTENCE_ENDS:
                        idx = i
                        break
                    if ch in _CLAUSE_ENDS and i >= _CLAUSE_MIN_LEN:
                        idx = i
                        break

                if idx == -1:
                    break

                sentence = buffer[:idx + 1].strip()
                buffer   = buffer[idx + 1:].lstrip()

                if sentence:
                    logger.debug(f"VLM stream yielding: '{sentence[:60]}'")
                    yield sentence

        # Flush any remaining text that didn't end with punctuation
        remainder = buffer.strip()
        if remainder:
            logger.debug(f"VLM stream flush: '{remainder[:60]}'")
            yield remainder

        total = time.time() - start
        logger.debug(f"VLM stream complete in {total:.2f}s")
