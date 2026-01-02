import uuid
import datetime
from backend.firebase_setup import db
from google.cloud import firestore

def create_interview(uid, questions, mode):
    sid = str(uuid.uuid4())
    data = {
        "mode": mode,
        "questions": questions,
        "idx": 0,
        "answers": [],
        "created_at": datetime.datetime.now().isoformat(),
        "status": "in_progress"
    }
    
    if db:
        # Save to Firestore: users/{uid}/interviews/{sid}
        doc_ref = db.collection("users").document(uid).collection("interviews").document(sid)
        doc_ref.set(data)
    else:
        # Fallback or error (for now we assume DB is up, or we could keep in-memory as fallback if needed)
        print("DB not initialized, skipping save")
        
    return sid

def next_question(uid, sid):
    if not db:
        return None
        
    doc_ref = db.collection("users").document(uid).collection("interviews").document(sid)
    doc = doc_ref.get()
    
    if not doc.exists:
        return None
        
    data = doc.to_dict()
    # Check bounds
    if data["idx"] >= len(data["questions"]):
        return None
        
    return data["questions"][data["idx"]]

def store_answer(uid, sid, q, a):
    if not db:
        return
        
    doc_ref = db.collection("users").document(uid).collection("interviews").document(sid)
    # Using array_union is simple, but we also want to increment idx.
    # To be safe transactionally we might read-write, but explicit update is fine for this MVP.
    
    # We want to append to 'answers' and increment 'idx'
    doc_ref.update({
        "answers": firestore.ArrayUnion([{"question": q, "answer": a}]),
        "idx": firestore.Increment(1)
    })

def get_answers(uid, sid):
    if not db:
        return []
        
    doc_ref = db.collection("users").document(uid).collection("interviews").document(sid)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("answers", [])
    return []

def get_user_analytics(uid):
    if not db:
        return []
    
    # Fetch all interviews
    docs = db.collection("users").document(uid).collection("interviews").order_by("created_at", direction=firestore.Query.DESCENDING).get()
    return [d.to_dict() | {"id": d.id} for d in docs]

def save_feedback(uid, sid, question, feedback_type, feedback_data):
    if not db:
        return

    doc_ref = db.collection("users").document(uid).collection("interviews").document(sid)
    
    # Transactional update not strictly required for this MVP, but read-modify-write needed for array item update
    doc = doc_ref.get()
    if not doc.exists:
        return
        
    data = doc.to_dict()
    answers = data.get("answers", [])
    
    updated = False
    for ans in answers:
        # Simple matching by question text
        if ans.get("question") == question:
            ans[feedback_type] = feedback_data
            updated = True
            break
            
    if updated:
        doc_ref.update({"answers": answers})
