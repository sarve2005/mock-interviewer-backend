from pydantic import BaseModel
from typing import Dict, List

class StartInterview(BaseModel):
    mode: str
    num_questions: int = 6

class QAInput(BaseModel):
    session_id: str
    question: str
    answer: str

class ScoreResponse(BaseModel):
    scores: Dict[str, int]

class SummaryResponse(BaseModel):
    summary: str

class FlagsResponse(BaseModel):
    flags: List[str]

class TTSInput(BaseModel):
    text: str
