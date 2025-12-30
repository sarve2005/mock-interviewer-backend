from backend.llm.gemini_client import generate_questions
from backend.llm.prompts import *
from backend.vectorstore.faiss_store import retrieve


def generate_questions_bulk(index, chunks, mode, n):
    query = "projects implementation" if mode == "technical" else "team leadership"
    context = "\n".join(retrieve(query, index, chunks, 8))
    
    questions = []
    for _ in range(n):
        prompt = technical_question_prompt(context, 1, questions) if mode == "technical" \
            else behavioural_question_prompt(context, 1, questions)

        raw = generate_questions(prompt)
        print(f"DEBUG: RAW QUESTION:\n{raw}")
        
        # Clean up the single question
        # It might still come with "1. " or similar, so we strip
        cleaned = raw.strip("1234567890. -*\"'\n")
        if cleaned:
            questions.append(cleaned)
            
    return questions
