from groq import Groq
from backend.config import GROQ_API_KEY
import os

client = Groq(api_key=GROQ_API_KEY)

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes the given audio file using Groq's Whisper model.
    """
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en",
                temperature=0.0
            )
            return transcription.text
    except Exception as e:
        print(f"Groq Transcription Error: {e}")
        raise e
