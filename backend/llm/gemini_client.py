from google import genai
from backend.config import *

_qgen = genai.Client(api_key=GEMINI_QGEN_API_KEY)
_scores = genai.Client(api_key=GEMINI_FEEDBACK_SCORES_API_KEY)
_summary = genai.Client(api_key=GEMINI_FEEDBACK_SUMMARY_API_KEY)
_flags = genai.Client(api_key=GEMINI_FEEDBACK_FLAGS_API_KEY)


def generate_questions(prompt: str) -> str:
    return _qgen.models.generate_content(
        model=GEMINI_QGEN_MODEL,
        contents=prompt
    ).text.strip()


def generate_scores(prompt: str) -> str:
    return _scores.models.generate_content(
        model=GEMINI_FEEDBACK_MODEL,
        contents=prompt
    ).text.strip()


def generate_summary(prompt: str) -> str:
    return _summary.models.generate_content(
        model=GEMINI_FEEDBACK_MODEL,
        contents=prompt
    ).text.strip()


def generate_flags(prompt: str) -> str:
    return _flags.models.generate_content(
        model=GEMINI_FEEDBACK_MODEL,
        contents=prompt
    ).text.strip()
