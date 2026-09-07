# OPTIONAL DEVELOPMENT FILE
# Stages 3–5: Assessment, Review, and Refinement.
# The deployable application is app.py.

from app import get_client, call_ai

def assessment_stage(topic, level, content):
    client = get_client()
    return call_ai(
        client,
        "You are an assessment agent. Create MCQs, short answers, applications, and an answer key.",
        f"Topic: {topic}\nLevel: {level}\nContent:\n{content}"
    )

def review_stage(topic, plan, content, assessment):
    client = get_client()
    return call_ai(
        client,
        "You are a quality-review agent. Find alignment, clarity, factual, and assessment issues.",
        f"PLAN:\n{plan}\nCONTENT:\n{content}\nASSESSMENT:\n{assessment}"
    )

def refinement_stage(topic, level, plan, content, assessment, review):
    client = get_client()
    return call_ai(
        client,
        "You are a final refinement agent. Produce a polished final study pack using all context.",
        f"Topic: {topic}\nLevel: {level}\nPLAN:\n{plan}\nCONTENT:\n{content}\nASSESSMENT:\n{assessment}\nREVIEW:\n{review}"
    )
