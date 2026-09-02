from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
)
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemma-4-26b-a4b-it:free"
)

# Lazy client — only instantiated on first get_llm() call so that
# importing this module in tests (without a real API key) does not crash.
_client: OpenAI | None = None


def get_llm() -> OpenAI:
    global _client
    if _client is None:
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file before running the full pipeline."
            )
        _client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    return _client


def get_model_name() -> str:
    return OPENROUTER_MODEL