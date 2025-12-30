from elevenlabs.client import ElevenLabs
from backend.config import ELEVENLABS_API_KEY
import io

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def transcribe_audio(file_obj) -> str:
    """
    Transcribes audio file-like object using ElevenLabs Scribe.
    """
    # ElevenLabs Python SDK's speech_to_text.convert requires a file-like object
    # or bytes. If it's bytes, we might need to wrap it.
    # Based on SDK docs, it usually expects a file path or file-like object with a name.
    
    # Ensure we are at the start of the stream
    if hasattr(file_obj, 'seek'):
        file_obj.seek(0)
    
    # Read content to bytes to avoid stream issues with the SDK
    content = file_obj.read()
    
    # Create a BytesIO object with a name, as some processing might check for .mp3/.wav extension
    # The SDK helper often expects a valid file-like with a name or a path.
    # We'll try passing bytes directly if SDK supports it, or wrap in BytesIO.
    # Looking at standard usage, passing a file-like object is best.
    
    # Let's verify what the SDK expects. It sends a multipart/form-data request.
    # We should probably pass it as a file-like with a 'name' attribute.
    
    result = client.speech_to_text.convert(
        file=io.BytesIO(content), # The SDK handles BytesIO, but we should inspect if it needs a name
        model_id="scribe_v1",
        tag_audio_events=False,
        language_code="eng"
    )
    
    return result.text
