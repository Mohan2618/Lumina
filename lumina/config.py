import os
from google import genai
import anthropic
from .prompts.system_prompt import SYSTEM_PROMPT

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_TIMEOUT = 30
ENABLE_CLAUDE = os.environ.get("ENABLE_CLAUDE", "false").lower() == "true"

GUEST_MSG_LIMIT = 5
GUEST_SESSION_KEY = "guest_messages"

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
claude_client = None
if ENABLE_CLAUDE and ANTHROPIC_API_KEY:
    try:
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception:
        claude_client = None
