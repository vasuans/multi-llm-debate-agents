"""
config.py

Loads environment variables and defines constants
for model names and global settings.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Set it in your .env file.")
if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY is missing. Set it in your .env file.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing. Set it in your .env file.")

# Low-cost / fast models (you can tweak these later)
OPENAI_DEBATER_MODEL = "gpt-4.1-mini"        # OpenAI debater (cheap)
GROK_DEBATER_MODEL = "grok-3-mini"           # Grok debater (xAI model) 
GEMINI_JUDGE_MODEL = "gemini-2.0-flash-lite" # Gemini judge

# Default creativity (temperature) range
DEFAULT_TEMPERATURE = 0.6
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.0
