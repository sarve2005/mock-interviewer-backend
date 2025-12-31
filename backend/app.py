from fastapi import FastAPI, UploadFile, File, Response
from fastapi.concurrency import run_in_threadpool
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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://mock-interviewer-frontend.vercel.app"
    ],
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
    
    # Run blocking PDF processing in threadpool
    text = await run_in_threadpool(resume_service.load_pdf_text, path)
    
    chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
    global vector_index, doc_chunks
    # Run heavy FAISS build in threadpool
    vector_index, doc_chunks = await run_in_threadpool(faiss_store.build_store, chunks)
    return {"status": "ok"}


@app.post("/start-interview")
async def start(req: StartInterview):
    # Generating questions involves LLM calls (blocking)
    qs = await run_in_threadpool(
        question_service.generate_questions_bulk,
        vector_index, doc_chunks, req.mode, req.num_questions
    )
    # Creating interview involves Firestore (blocking)
    sid = await run_in_threadpool(interview_service.create_interview, qs, req.mode)
    return {"session_id": sid}


@app.get("/next-question/{sid}")
async def next_q(sid: str):
    # Firestore read
    q = await run_in_threadpool(interview_service.next_question, sid)
    return {"question": q}

@app.post("/submit-answer")
async def submit_ans(d: AnswerInput):
    # Firestore write
    await run_in_threadpool(interview_service.store_answer, d.session_id, d.question, d.answer)
    return {"status": "ok"}



@app.post("/technical/scores", response_model=ScoreResponse)
async def tech_scores(d: QAInput):
    scores = await run_in_threadpool(feedback_service.technical_scores, d.question, d.answer)
    return {"scores": scores}

@app.post("/technical/summary", response_model=SummaryResponse)
async def tech_summary(d: QAInput):
    summary = await run_in_threadpool(feedback_service.technical_summary, d.question, d.answer)
    return {"summary": summary}


@app.post("/technical/flags", response_model=FlagsResponse)
async def tech_flags(d: QAInput):
    flags = await run_in_threadpool(feedback_service.technical_flags, d.question, d.answer)
    return {"flags": flags}


@app.post("/tts")
async def tts_endpoint(d: TTSInput):
    # Network call to ElevenLabs/GTTS
    audio = await run_in_threadpool(tts_service.text_to_speech_bytes, d.text)
    return Response(content=audio, media_type="audio/mpeg")


@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    # UploadFile.file is a SpooledTemporaryFile which is file-like
    # This might be tricky if stt_service expects a path or modifies the file pointer
    # But generally assuming it reads from the file-like object
    text = await run_in_threadpool(stt_service.transcribe_audio, file.file)
    return {"text": text}


@app.get("/session/{sid}")
async def get_session(sid: str):
    answers = await run_in_threadpool(interview_service.get_answers, sid)
    # We might want to return more info if needed
    return {"answers": answers}

