import uuid
import datetime

# In-memory storage (Replacement for Firestore)
_SESSIONS = {}

def create_interview(questions, mode):
    sid = str(uuid.uuid4())
    data = {
        "mode": mode,
        "questions": questions,
        "idx": 0,
        "answers": [],
        "created_at": datetime.datetime.now().isoformat()
    }
    
    _SESSIONS[sid] = data
    return sid


def next_question(sid):
    if sid not in _SESSIONS:
        return None
    s = _SESSIONS[sid]
    if s["idx"] >= len(s["questions"]):
        return None
    return s["questions"][s["idx"]]


def store_answer(sid, q, a):
    if sid in _SESSIONS:
        _SESSIONS[sid]["answers"].append({"question": q, "answer": a})
        _SESSIONS[sid]["idx"] += 1


def get_answers(sid):
    if sid in _SESSIONS:
        return _SESSIONS[sid]["answers"]
    return []
