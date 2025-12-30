import uuid
from backend.firebase_config import get_db

# Fallback in-memory storage if Firebase fails or not configured
_SESSIONS = {}

def create_interview(questions, mode):
    sid = str(uuid.uuid4())
    data = {
        "mode": mode,
        "questions": questions,
        "idx": 0,
        "answers": [],
        "created_at": firestore.SERVER_TIMESTAMP if 'firestore' in globals() else None
    }
    
    db = get_db()
    if db:
        try:
            db.collection("sessions").document(sid).set(data)
            return sid
        except Exception as e:
            print(f"Firestore create error: {e}")
    
    # Fallback
    _SESSIONS[sid] = data
    return sid


def next_question(sid):
    db = get_db()
    if db:
        try:
            doc = db.collection("sessions").document(sid).get()
            if doc.exists:
                data = doc.to_dict()
                if data["idx"] >= len(data["questions"]):
                    return None
                
                return data["questions"][data["idx"]]
        except Exception as e:
            print(f"Firestore next_question error: {e}")

    # Fallback
    if sid not in _SESSIONS:
        return None
    s = _SESSIONS[sid]
    if s["idx"] >= len(s["questions"]):
        return None
    return s["questions"][s["idx"]]


def store_answer(sid, q, a):
    db = get_db()
    if db:
        try:
            entry = {"question": q, "answer": a}
            
            doc_ref = db.collection("sessions").document(sid)
            # Use firestore.ArrayUnion
            from google.cloud import firestore as google_firestore
            
            # Transactional update would be better, but for now:
            # 1. Add answer
            # 2. Increment idx
            doc_ref.update({
                "answers": google_firestore.ArrayUnion([entry]),
                "idx": google_firestore.Increment(1)
            })
            return
        except Exception as e:
            print(f"Firestore store_answer error: {e}")

    # Fallback
    if sid in _SESSIONS:
        _SESSIONS[sid]["answers"].append({"question": q, "answer": a})
        _SESSIONS[sid]["idx"] += 1


def get_answers(sid):
    db = get_db()
    if db:
        try:
            doc = db.collection("sessions").document(sid).get()
            if doc.exists:
                return doc.to_dict().get("answers", [])
        except Exception as e:
            print(f"Firestore get_answers error: {e}")
            
    if sid in _SESSIONS:
        return _SESSIONS[sid]["answers"]
    return []
