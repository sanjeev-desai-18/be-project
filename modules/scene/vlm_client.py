# # modules/scene/vlm_client.py — Vision client using Groq's vision model.
# # Groq supports llama-3.2-11b-vision-preview for image understanding.

# from groq import Groq
# from utils.logger import logger
# from config import GROQ_API_KEY, VLM_MODEL, VLM_MAX_TOKENS


# class VLMClient:
#     """
#     Sends image + prompt to Groq Vision model.
#     Used by scene, reading, and currency modules.
#     """

#     def __init__(self):
#         self.client = Groq(api_key=GROQ_API_KEY)
#         self.model  = VLM_MODEL
#         logger.debug(f"VLMClient ready — model: {self.model}")

#     def describe(self, image_b64: str, prompt: str) -> str:
#         """
#         Send base64 image + text prompt to Groq Vision.

#         Args:
#             image_b64: base64-encoded JPEG string
#             prompt:    instruction for what to do with the image

#         Returns:
#             Text response. Returns error message string on failure.
#         """
#         logger.debug(f"Calling Groq Vision ({self.model})...")

#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 max_tokens=VLM_MAX_TOKENS,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/jpeg;base64,{image_b64}"
#                                 }
#                             },
#                             {
#                                 "type": "text",
#                                 "text": prompt
#                             }
#                         ]
#                     }
#                 ]
#             )

#             result = response.choices[0].message.content.strip()
#             logger.debug(f"VLM response: '{result[:100]}'")
#             return result

#         except Exception as e:
#             logger.error(f"Groq Vision API call failed: {e}")
#             return "I was unable to analyse the image right now. Please try again."

# modules/scene/vlm_client.py

from groq import Groq
from utils.logger import logger
from config import GROQ_API_KEY, VLM_MODEL, VLM_MAX_TOKENS
import time


class VLMClient:
    """
    Sends image + prompt to Groq Vision model.
    Used by scene, reading, and currency modules.
    """

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model  = VLM_MODEL
        logger.debug(f"VLMClient ready — model: {self.model}")

    def describe(self, image_b64: str, prompt: str) -> str:

        logger.debug(f"Calling Groq Vision ({self.model})...")

        start = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=VLM_MAX_TOKENS,
                timeout=30,   # prevents long blocking
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            latency = time.time() - start
            logger.debug(f"VLM latency: {latency:.2f}s")

            result = response.choices[0].message.content if response.choices else ""
            result = result.strip()

            logger.debug(f"VLM response: '{result[:100]}'")

            return result

        except Exception as e:
            logger.error(f"Groq Vision API call failed: {e}")
            return "I was unable to analyse the image right now. Please try again."