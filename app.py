import os
import time
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="📚",
    layout="wide",
)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_MODEL = "openai/gpt-oss-20b"

# Conservative limits to help keep requests below the Groq TPM limit.
MAX_PLAN_CHARS = 2200
MAX_CONTENT_CHARS = 5000
MAX_ASSESSMENT_CHARS = 3000
MAX_REVIEW_CHARS = 1800


def get_client():
    """Load GROQ_API_KEY from Streamlit Secrets first, then environment."""
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets or your environment."
        )

    return Groq(api_key=api_key)


def limit_text(text, max_chars):
    """Keep context small enough for multi-stage API requests."""
    if not text:
        return ""

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    # Keep the beginning and end because both often contain useful context.
    head_size = int(max_chars * 0.75)
    tail_size = max_chars - head_size

    return (
        text[:head_size]
        + "\n\n[... context shortened to reduce API request size ...]\n\n"
        + text[-tail_size:]
    )


def call_ai(
    client,
    system_prompt,
    user_prompt,
    model=DEFAULT_MODEL,
    temperature=0.4,
    max_tokens=1000,
    retries=2,
    system = """You are the Assessment Agent.
Create a balanced assessment from the supplied study content.
Include 8 multiple-choice questions, 4 short-answer questions, and
2 application questions. Put an answer key at the end. Do not assess
material that is not reasonably covered by the content."""
    prompt = f"""
Topic: {topic}
Level: {level}

STUDY CONTENT:
{content}

Generate the assessment and answer key.
"""
    return call_ai(client, system, prompt, temperature=0.3)

def stage_review(client, topic, plan, content, assessment):

    system = """You are the Review and Quality Agent.

Review the study materials for:

1. Alignment with the topic
2. Factual consistency
3. Missing objectives
4. Unclear explanations
5. Duplicated questions
6. Mismatches between content and assessment

Return ONLY:

## Strengths
## Issues Found
## Required Fixes
## Quality Score

Keep the review concise and under 600 words.
Do not rewrite the entire study pack.
"""

    prompt = f"""
Topic: {topic}

PLAN:
{limit_text(plan, 2000)}

CONTENT:
{limit_text(content, 4000)}

ASSESSMENT:
{limit_text(assessment, 2500)}

Review these materials.
"""

    return call_ai(
        client,
        system,
        prompt,
        temperature=0.2,
        max_tokens=800
    )

# -----------------------------
# Safe stage runner / error handling
# -----------------------------
def run_stage(name, fn, *args):
    try:
        with st.status(f"{name} in progress...", expanded=False):
            result = fn(*args)
        if not result:
            raise RuntimeError("The AI returned an empty response.")
        return result, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

# -----------------------------
# UI
# -----------------------------
st.title("📚 AI Study Pack Generator")
st.caption("Multi-stage AI workflow: Planning → Content → Assessment → Review → Refinement")

with st.sidebar:
    st.header("Student Profile")
    topic = st.text_input("Study topic", placeholder="e.g., Machine Learning")
    level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
    duration = st.number_input("Duration (weeks)", min_value=1, max_value=52, value=4)
    hours = st.number_input("Study hours/day", min_value=0.5, max_value=12.0, value=2.0, step=0.5)
    goals = st.text_area("Learning goals", placeholder="What do you want to achieve?")
    generate = st.button("🚀 Generate Study Pack", type="primary", use_container_width=True)

if generate:
    if not topic.strip():
        st.error("Please enter a study topic.")
        st.stop()

    try:
        client = get_client()
    except Exception as exc:
        st.error(str(exc))
        st.info("For Streamlit Cloud, add GROQ_API_KEY under Settings → Secrets.")
        st.stop()

    st.session_state.clear()

    # 1. Planning
    plan, err = run_stage(
        "Stage 1 — Planning",
        stage_planning, client, topic, level, duration, hours, goals
    )
    if err:
        st.error(f"Planning failed: {err}")
        st.stop()
    st.session_state["plan"] = plan

    # 2. Content generation; planning context is passed forward
    content, err = run_stage(
        "Stage 2 — Content Generation",
        stage_content, client, topic, level, plan
    )
    if err:
        st.error(f"Content generation failed: {err}")
        st.stop()
    st.session_state["content"] = content

    # 3. Assessment; content context is passed forward
    assessment, err = run_stage(
        "Stage 3 — Assessment",
        stage_assessment, client, topic, level, content
    )
    if err:
        st.error(f"Assessment failed: {err}")
        st.stop()
    st.session_state["assessment"] = assessment

    # 4. Review; all relevant context is passed forward
    review, err = run_stage(
        "Stage 4 — Review & Quality Check",
        stage_review, client, topic, plan, content, assessment
    )
    if err:
        st.error(f"Review failed: {err}")
        st.stop()
    st.session_state["review"] = review

    # 5. Refinement; full context is passed to final agent
    def stage_refinement(client, topic, level, plan, content, assessment, review):

    system = """You are the Final Refinement Agent.

Create a polished and concise final study pack.

Use these Markdown sections:

# Study Roadmap
# Learning Objectives
# Study Notes
# Quick Revision
# Practice Questions
# Answer Key
# Final Review Tips

Use the quality review to improve the material.
Do not mention that you are an AI.
Keep the final study pack concise.
"""

    prompt = f"""
Topic: {topic}
Level: {level}

PLAN SUMMARY:
{limit_text(plan, 1500)}

STUDY CONTENT:
{limit_text(content, 4500)}

ASSESSMENT:
{limit_text(assessment, 2500)}

QUALITY REVIEW:
{limit_text(review, 2000)}
"""

    return call_ai(
        client,
        system,
        prompt,
        temperature=0.3,
        max_tokens=1500
    )
