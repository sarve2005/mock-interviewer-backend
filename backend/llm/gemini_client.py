from google import genai
from backend.config import *

_qgen = None
_scores = None
_summary = None
_flags = None

def get_client(api_key):
    # Lazy init to prevent startup crashes if keys are missing
    if not api_key:
        print("Warning: Gemini API Key not set.")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return None

def generate_questions(prompt: str) -> str:
    global _qgen
    if _qgen is None:
        _qgen = get_client(GEMINI_QGEN_API_KEY)
    
    if not _qgen:
        return "Error: Gemini API Key missing or invalid."

    return _qgen.models.generate_content(
        model=GEMINI_QGEN_MODEL,
        contents=prompt
    ).text.strip()


def generate_scores(prompt: str) -> str:
    global _scores
    if _scores is None:
        _scores = get_client(GEMINI_FEEDBACK_SCORES_API_KEY)

    if not _scores:
        return "Error: Gemini API Key missing."

    return _scores.models.generate_content(
        model=GEMINI_FEEDBACK_MODEL,
        contents=prompt
    ).text.strip()


def generate_summary(prompt: str) -> str:
    global _summary
    if _summary is None:
        _summary = get_client(GEMINI_FEEDBACK_SUMMARY_API_KEY)

    if not _summary:
        return "Error: Gemini API Key missing."

    return _summary.models.generate_content(
        model=GEMINI_FEEDBACK_MODEL,
        contents=prompt
    ).text.strip()


def generate_flags(prompt: str) -> str:
    global _flags
    # Assuming flags uses the same key as scores/summary if not defined, 
    # but here sticking to config variable.
    if _flags is None:
        _flags = get_client(GEMINI_FEEDBACK_FLAGS_API_KEY)

    if not _flags:
        return "Error: Gemini API Key missing."

    return _flags.models.generate_content(
        model=GEMINI_FEEDBACK_MODEL,
        contents=prompt
    ).text.strip()
