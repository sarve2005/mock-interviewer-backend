from elevenlabs.client import ElevenLabs
from backend.config import ELEVENLABS_API_KEY

client = None

def get_client():
    global client
    if not client:
        if not ELEVENLABS_API_KEY:
             print("Warning: ElevenLabs API Key not set.")
             return None
        try:
            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        except Exception as e:
            print(f"Error initializing ElevenLabs client: {e}")
            return None
    return client

def text_to_speech_bytes(text: str) -> bytes:
    c = get_client()
    if not c:
        return b"" # Or raise error
    
    audio = c.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    return b"".join(audio)
