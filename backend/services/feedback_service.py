import re
from backend.llm.gemini_client import *

def _extract_scores(txt):
    scores = {}
    # Use regex to find "Key: Value" patterns, handling optional asterisks or spaces
    matches = re.findall(r"(?:\*+)?([a-zA-Z]+)(?:\*+)?\s*:\s*(\d+)", txt)
    for key, val in matches:
        scores[key] = int(val)
    return scores

def technical_scores(q, a):
    prompt = f"""
Analyze the following technical interview question and answer.
Provide scores (1-5) for: Concept, Depth, Reasoning, Clarity, Confidence.

Q: {q}
A: {a}

Output format:
Concept: <int>
Depth: <int>
Reasoning: <int>
Clarity: <int>
Confidence: <int>
"""
    return _extract_scores(generate_scores(prompt))


def technical_summary(q, a):
    return generate_summary(f"Summarize:\nQ:{q}\nA:{a}")


ALLOWED_FLAGS = {"OFF_TOPIC", "INCORRECT", "VAGUE", "TOO_SHORT", "MINOR_ERROR"}

def technical_flags(q, a):
    prompt = f"""
Analyze the answer. If there are issues, output one or more of these flags:
{", ".join(ALLOWED_FLAGS)}

If the answer is acceptable/perfect or just a normal answer, output: NONE

Q: {q}
A: {a}

Output format:
FLAGS: <comma_separated_flags_or_NONE>
"""
    raw = generate_flags(prompt)
    
    # Extract flags using regex but filter by ALLOWED_FLAGS
    found_flags = re.findall(r"[A-Z_]{3,}", raw)
    valid_flags = [f for f in found_flags if f in ALLOWED_FLAGS]
    
    return valid_flags
