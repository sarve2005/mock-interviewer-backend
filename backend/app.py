from fastapi import FastAPI, UploadFile, File, Response, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from backend.services import resume_service, question_service, interview_service, feedback_service
from backend.vectorstore import faiss_store
from backend.models.schemas import *
from pydantic import BaseModel
from contextlib import asynccontextmanager
from backend.auth import verify_token

class AnswerInput(BaseModel):
    session_id: str
    question: str
    answer: str

    session_id: str
    question: str
    answer: str

import shutil, os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the generic embedding model to avoid first-request latency
    print("Preloading FastEmbed model...")
    # This might block, but it's acceptable during startup
    await run_in_threadpool(faiss_store.get_embedder)
    print("FastEmbed model loaded successfully.")
    yield

app = FastAPI(lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://mock-interviewer-frontend.vercel.app",
        "https://mock-interviewer-77229.web.app"
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
async def start(req: StartInterview, uid: str = Depends(verify_token)):
    # Generating questions involves LLM calls (blocking)
    qs = await run_in_threadpool(
        question_service.generate_questions_bulk,
        vector_index, doc_chunks, req.mode, req.num_questions
    )
    # Creating interview involves Firestore (blocking)
    sid = await run_in_threadpool(interview_service.create_interview, uid, qs, req.mode)
    return {"session_id": sid}


@app.get("/next-question/{sid}")
async def next_q(sid: str, uid: str = Depends(verify_token)):
    q = await run_in_threadpool(interview_service.next_question, uid, sid)
    return {"question": q}

@app.post("/submit-answer")
async def submit_ans(d: AnswerInput, uid: str = Depends(verify_token)):
    await run_in_threadpool(interview_service.store_answer, uid, d.session_id, d.question, d.answer)
    return {"status": "ok"}



@app.post("/technical/scores", response_model=ScoreResponse)
async def tech_scores(d: QAInput, uid: str = Depends(verify_token)):
    scores = await run_in_threadpool(feedback_service.technical_scores, d.question, d.answer)
    # Persist scores
    await run_in_threadpool(interview_service.save_feedback, uid, d.session_id, d.question, "scores", scores)
    return {"scores": scores}

@app.post("/technical/summary", response_model=SummaryResponse)
async def tech_summary(d: QAInput, uid: str = Depends(verify_token)):
    summary = await run_in_threadpool(feedback_service.technical_summary, d.question, d.answer)
    # Persist summary (optional, but good for history)
    await run_in_threadpool(interview_service.save_feedback, uid, d.session_id, d.question, "feedback_summary", summary)
    return {"summary": summary}


@app.post("/technical/flags", response_model=FlagsResponse)
async def tech_flags(d: QAInput, uid: str = Depends(verify_token)):
    flags = await run_in_threadpool(feedback_service.technical_flags, d.question, d.answer)
    # Persist flags
    await run_in_threadpool(interview_service.save_feedback, uid, d.session_id, d.question, "flags", flags)
    return {"flags": flags}


@app.get("/session/{sid}")
async def get_session(sid: str, uid: str = Depends(verify_token)):
    answers = await run_in_threadpool(interview_service.get_answers, uid, sid)
    # We might want to return more info if needed
    return {"answers": answers}

@app.get("/analytics")
async def get_analytics(uid: str = Depends(verify_token)):
    data = await run_in_threadpool(interview_service.get_user_analytics, uid)
    return {"history": data}


@app.post("/transcribe")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """
    Receives an audio file (webm/wav), saves it temporarily, and transcribes it using Groq.
    """
    try:
        # Create a temp file
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        # Transcribe
        from backend.services.transcription_service import transcribe_audio
        text = transcribe_audio(temp_filename)
        
        # Cleanup
        os.remove(temp_filename)
        
        return {"text": text}
    except Exception as e:
        print(f"Transcription error: {e}")
        # Cleanup if exists
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))

