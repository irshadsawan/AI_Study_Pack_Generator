# OPTIONAL DEVELOPMENT FILE
# Stages 1–2: Planning and Content Generation.
# The deployable application is app.py.

from app import get_client, call_ai

def planning_stage(topic, level, duration, hours, goals):
    client = get_client()
    return call_ai(
        client,
        "You are a study planning agent. Build a personalized learning roadmap.",
        f"Topic: {topic}\nLevel: {level}\nDuration: {duration} weeks\nHours/day: {hours}\nGoals: {goals}"
    )

def content_stage(topic, level, plan):
    client = get_client()
    return call_ai(
        client,
        "You are a study content agent. Create accurate, level-appropriate notes.",
        f"Topic: {topic}\nLevel: {level}\nPlanning context:\n{plan}"
    )
