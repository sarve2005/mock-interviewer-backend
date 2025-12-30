import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def analyze_audio_answer(*args, **kwargs):
    raise NotImplementedError("Multimodal is disabled.")