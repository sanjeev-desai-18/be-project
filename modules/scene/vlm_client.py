# modules/scene/vlm_client.py
#
# Changes from previous version:
#   1. _CLAUSE_ENDS (comma / semicolon splits) removed entirely.
#      They produced mid-sentence fragments like "There is a chair,"
#      which gTTS renders with an awkward hang before the next call.
#      Only true sentence-end punctuation (.!?\n) triggers a yield.
#
#   2. Remainder flush: if the VLM finishes without a terminal punctuation
#      mark (common — models often drop the final period), a "." is appended
#      before yielding so gTTS synthesises a clean, complete-sounding phrase
#      instead of cutting off the last phoneme.

import time
from groq import Groq
from utils.logger import logger
from config import GROQ_API_KEY, VLM_MODEL, VLM_MAX_TOKENS


# Only yield on true sentence boundaries — never on commas or semicolons.
# Comma splits produce tiny fragments that gTTS renders unnaturally.
_SENTENCE_ENDS = frozenset({".", "!", "?", "\n"})


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

        Splits ONLY on true sentence-end punctuation (.!?\\n).
        Never splits on commas or semicolons — those produce fragments that
        gTTS renders with an unnatural hanging pause.

        Remainder handling: if the stream ends without a terminal punctuation
        mark, a period is appended before yielding so gTTS generates a clean,
        fully-voiced phrase rather than cutting off the last phoneme.

        Yields:
            str — one complete sentence at a time, stripped of whitespace.
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

            # Drain all complete sentences from the front of the buffer.
            # Only sentence-end punctuation triggers a yield — no comma splits.
            while True:
                idx = -1
                for i, ch in enumerate(buffer):
                    if ch in _SENTENCE_ENDS:
                        idx = i
                        break

                if idx == -1:
                    break

                sentence = buffer[:idx + 1].strip()
                buffer   = buffer[idx + 1:].lstrip()

                if sentence:
                    logger.debug(f"VLM stream yielding: '{sentence[:80]}'")
                    yield sentence

        # Flush any remaining text that did not end with punctuation.
        # Append "." so gTTS voices the last phoneme cleanly instead of
        # cutting off mid-word.
        remainder = buffer.strip()
        if remainder:
            if remainder[-1] not in _SENTENCE_ENDS:
                remainder += "."
            logger.debug(f"VLM stream flush: '{remainder[:80]}'")
            yield remainder

        total = time.time() - start
        logger.debug(f"VLM stream complete in {total:.2f}s")
