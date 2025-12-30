def technical_question_prompt(context, n, excluded=None):
    exclude_str = ""
    if excluded:
        exclude_str = f"Do NOT repeat these questions: {excluded}"
        
    return f"""
Generate EXACTLY {n} technical interview question.
{exclude_str}
Resume:
{context}
"""


def behavioural_question_prompt(context, n, excluded=None):
    exclude_str = ""
    if excluded:
        exclude_str = f"Do NOT repeat these questions: {excluded}"

    return f"""
Generate EXACTLY {n} behavioural interview questions using STAR.
{exclude_str}
Resume:
{context}
"""
