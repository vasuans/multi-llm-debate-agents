"""
clients.py

Thin wrappers around:
- OpenAI (Debater A)
- Grok via xAI API (Debater B)
- Gemini (Judge)

We keep functions simple so it's easy to read and debug.
"""

from typing import List, Dict

from openai import OpenAI
import google.generativeai as genai

from .config import (
    OPENAI_API_KEY,
    GROK_API_KEY,
    GEMINI_API_KEY,
    OPENAI_DEBATER_MODEL,
    GROK_DEBATER_MODEL,
    GEMINI_JUDGE_MODEL,
)

# -------------------------------
# Initialize API clients
# -------------------------------

# Regular OpenAI client for Debater A + embeddings
openai_debater_client = OpenAI(api_key=OPENAI_API_KEY)

# xAI Grok API is OpenAI-compatible: just change base_url to https://api.x.ai/v1 
grok_client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1",
)

# Gemini uses its own style of client
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_JUDGE_MODEL)


# -------------------------------
# Helper functions
# -------------------------------

def call_openai_debater(
    messages: List[Dict[str, str]],
    temperature: float = 0.6,
    max_tokens: int = 220,
) -> str:
    """
    Call OpenAI for Debater A.
    Input: list of messages (system/user/assistant).
    Output: text content only.
    """
    resp = openai_debater_client.chat.completions.create(
        model=OPENAI_DEBATER_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def call_grok_debater(
    messages: List[Dict[str, str]],
    temperature: float = 0.6,
    max_tokens: int = 220,
) -> str:
    """
    Call Grok (xAI) for Debater B using the OpenAI-compatible API.
    """
    resp = grok_client.chat.completions.create(
        model=GROK_DEBATER_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def call_gemini_judge(
    prompt_text: str,
    temperature: float = 0.3,
    max_tokens: int = 220,
) -> str:
    """
    Call Gemini 2.0 Flash Lite for the Judge.

    We send a single combined text prompt instead of chat messages
    to keep it simple.
    """
    response = gemini_model.generate_content(
        prompt_text,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
    )
    # response.text is the main text generation
    return (response.text or "").strip()
