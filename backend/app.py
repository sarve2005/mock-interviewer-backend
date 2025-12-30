from fastapi import FastAPI, UploadFile, File, Response
from backend.services import resume_service, question_service, interview_service, feedback_service, tts_service, stt_service
from backend.vectorstore import faiss_store
from backend.models.schemas import *
from pydantic import BaseModel

class AnswerInput(BaseModel):
    session_id: str
    question: str
    answer: str

import shutil, os

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, or configure from env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS = "uploads"
os.makedirs(UPLOADS, exist_ok=True)

vector_index = None
doc_chunks = None

@app.post("/upload-resume")
async def upload(file: UploadFile = File(...)):
    path = f"{UPLOADS}/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    text = resume_service.load_pdf_text(path)
    chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
    global vector_index, doc_chunks
    vector_index, doc_chunks = faiss_store.build_store(chunks)
    return {"status": "ok"}


@app.post("/start-interview")
async def start(req: StartInterview):
    qs = question_service.generate_questions_bulk(
        vector_index, doc_chunks, req.mode, req.num_questions
    )
    sid = interview_service.create_interview(qs, req.mode)
    return {"session_id": sid}


@app.get("/next-question/{sid}")
async def next_q(sid: str):
    q = interview_service.next_question(sid)
    return {"question": q}

@app.post("/submit-answer")
async def submit_ans(d: AnswerInput):
    interview_service.store_answer(d.session_id, d.question, d.answer)
    return {"status": "ok"}



@app.post("/technical/scores", response_model=ScoreResponse)
async def tech_scores(d: QAInput):
    return {"scores": feedback_service.technical_scores(d.question, d.answer)}

@app.post("/technical/summary", response_model=SummaryResponse)
async def tech_summary(d: QAInput):
    return {"summary": feedback_service.technical_summary(d.question, d.answer)}


@app.post("/technical/flags", response_model=FlagsResponse)
async def tech_flags(d: QAInput):
    return {"flags": feedback_service.technical_flags(d.question, d.answer)}


@app.post("/tts")
async def tts_endpoint(d: TTSInput):
    audio = tts_service.text_to_speech_bytes(d.text)
    return Response(content=audio, media_type="audio/mpeg")


@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    # UploadFile.file is a SpooledTemporaryFile which is file-like
    text = stt_service.transcribe_audio(file.file)
    return {"text": text}


@app.get("/session/{sid}")
async def get_session(sid: str):
    answers = interview_service.get_answers(sid)
    # We might want to return more info if needed
    return {"answers": answers}

