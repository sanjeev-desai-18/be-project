# # modules/scene/prompt_templates.py
# # Smart adaptive prompt — handles scene, object queries, and spatial queries.

# # Keywords that suggest user is asking about specific objects
# OBJECT_KEYWORDS = [
#     "what is", "what's", "is there", "any", "find", "locate", "where is",
#     "can you see", "do you see", "tell me about", "show me",
#     "kya hai", "kuch hai", "dekho", "batao", "kya dikh",
#     "chair", "table", "door", "window", "person", "people", "who",
#     "bottle", "bag", "phone", "laptop", "book", "food", "paisa",
#     "note", "pen", "cup", "glass", "key", "wallet", "clothes",
#     "holding", "hand", "left", "right", "front", "behind", "above", "below",
#     "near", "close", "far", "next to", "beside", "on top"
# ]

# # Spatial direction mappings — user's perspective → image region
# SPATIAL_HINTS = {
#     "left hand":     "Look at the LEFT side of the image, specifically at any hand visible there.",
#     "right hand":    "Look at the RIGHT side of the image, specifically at any hand visible there.",
#     "left":          "Focus on the LEFT portion of the image.",
#     "right":         "Focus on the RIGHT portion of the image.",
#     "front":         "Focus on what is directly ahead — CENTER of the image.",
#     "above":         "Look at the UPPER portion of the image.",
#     "below":         "Look at the LOWER portion of the image.",
#     "behind":        "This may not be visible in the frame — describe what you can see.",
#     "in my hand":    "Look for any hands in the image and identify what they are holding.",
#     "holding":       "Look for any hands in the image and identify what they are holding.",
#     "near me":       "Focus on objects closest to the camera in the foreground.",
#     "next to me":    "Focus on objects in the immediate foreground or sides.",
# }


# def detect_spatial_context(user_query: str, extra_context: str) -> str:
#     """
#     Extract spatial instructions from user query and extra_context.
#     Returns a specific instruction string for the VLM.
#     """
#     combined = (user_query + " " + extra_context).lower()
#     for key, instruction in SPATIAL_HINTS.items():
#         if key in combined:
#             return instruction
#     return ""


# def is_object_query(user_query: str) -> bool:
#     """Returns True if user is asking about specific objects."""
#     query_lower = user_query.lower()
#     return any(kw in query_lower for kw in OBJECT_KEYWORDS)


# def get_scene_prompt(extra_context: str = "", user_query: str = "") -> str:
#     """
#     Build adaptive VLM prompt based on what the user asked.

#     Three prompt types:
#     1. Pure scene     → general context, no object lists
#     2. Spatial query  → focused on specific region/hand/direction
#     3. Object query   → identify specific objects with location
#     """
#     spatial_instruction = detect_spatial_context(user_query, extra_context)
#     object_query        = is_object_query(user_query)
#     query_lower         = user_query.lower()

#     # ── TYPE 1: Spatial / Hand Query ──────────────────
#     # "what is in my left hand", "what am I holding", "what is on my right"
#     is_spatial = bool(spatial_instruction) or any(
#         kw in query_lower for kw in [
#             "hand", "holding", "left", "right", "front", "above",
#             "below", "behind", "near me", "next to"
#         ]
#     )

#     if is_spatial:
#         return f"""
# You are assisting a visually impaired person.
# They asked: "{user_query}"

# {spatial_instruction}

# Your task:
# 1. Look carefully at the specific area or region mentioned
# 2. Identify exactly what object, item, or thing is there
# 3. If it is a hand — identify what the hand is holding, even if partially visible
# 4. Give the answer directly and confidently

# RULES:
# - Answer the question DIRECTLY in the first sentence
# - Be specific — name the object clearly (e.g. "a blue pen", "a mobile phone", "a 500 rupee note")
# - If you genuinely cannot see that area or it is unclear, say so honestly
# - Use natural directions: "in your left hand", "on your right side", "in front of you"
# - No bullet points — spoken aloud
# - Max 3 sentences total

# GOOD: "You appear to be holding a mobile phone in your left hand."
# GOOD: "There is a water bottle on your right side on the table."
# BAD:  "I can see a hand, phone, table, cup, and some other items."
# """.strip()

#     # ── TYPE 2: Specific Object Query ─────────────────
#     # "is there a chair?", "can you see a door?", "where is the exit?"
#     elif object_query:
#         direction_line = f"\nUser context: '{extra_context}'" if extra_context else ""
#         return f"""
# You are assisting a visually impaired person.
# They asked: "{user_query}"{direction_line}

# Your task:
# 1. Directly answer whether the object/thing they asked about is visible
# 2. If yes — say where it is relative to the person (left, right, front, nearby)
# 3. Then briefly describe the overall environment in one sentence

# RULES:
# - Answer their question FIRST and DIRECTLY
# - Use positional language: "to your left", "directly ahead", "behind you", "nearby"
# - If the object is not visible, say clearly: "I cannot see a [object] from this view"
# - No bullet points — spoken aloud
# - Max 3 sentences

# GOOD: "Yes, there is a door directly ahead of you, slightly to the left."
# GOOD: "I cannot see a water bottle from this angle."
# BAD:  "I see a door, chair, table, window, lamp, and a bag."
# """.strip()

#     # ── TYPE 3: Pure Scene Description ────────────────
#     # "describe my surroundings", "where am I?"
#     else:
#         direction_line = f"\nUser mentioned: '{extra_context}'. Focus there." if extra_context else ""
#         return f"""
# You are a scene narrator for a visually impaired person.
# Describe the scene in 2-3 short, natural spoken sentences.{direction_line}

# Structure:
# 1. Setting  — What kind of environment is this?
# 2. Situation — What is generally happening or the mood/activity?
# 3. Alert    — ONLY if there is an immediate hazard. Skip if nothing urgent.

# RULES:
# - Do NOT list objects randomly
# - Do NOT say "I see", "I notice", "There is a"
# - Speak naturally as if guiding someone who cannot see
# - No bullet points — spoken aloud
# - Max 4 sentences

# GOOD: "You are in a quiet home workspace. It feels calm with moderate natural light coming in."
# BAD:  "I see a chair, table, lamp, laptop, bottle, window, curtain, and a bag."
# """.strip()


def get_scene_reasoning_prompt(scene_data: dict, user_query: str) -> str:
    return f"""
You are assisting a visually impaired person.

Scene awareness:
{scene_data}

User said:
{user_query}

Respond naturally with only relevant information.
Prioritize safety and objects the user interacts with.
""".strip()
