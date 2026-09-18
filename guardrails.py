
def guardrail(question):
    if not question or not question.strip():
        return False, "Please provide a question."

    if len(question) > 1000:
        return False, "Question is too long."

    blocked_terms = [
        "ignore previous instructions",
        "reveal system prompt",
        "show your system prompt"
    ]

    q = question.lower()

    for term in blocked_terms:
        if term in q:
            return False, "I can't follow that instruction."

    return True, ""
