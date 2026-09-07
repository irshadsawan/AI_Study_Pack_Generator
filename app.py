import os
import time
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="AI Study Pack Generator",
    page_icon="",
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


def call_ai(client, system_prompt, user_prompt, model=DEFAULT_MODEL, temperature=0.4, max_tokens=1000, retries=2):
    """Call Groq with conservative output limits and retry empty responses."""
    last_error = None

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort="low",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            if not response.choices:
                raise RuntimeError("The AI API returned no response choices.")

            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "content", None)

            if content and str(content).strip():
                return str(content).strip()

            finish_reason = getattr(choice, "finish_reason", "unknown")
            last_error = RuntimeError(
                f"The AI returned an empty response (finish reason: {finish_reason})."
            )

        except Exception as exc:
            last_error = exc

        # Short retry delay. Do not retry forever.
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(str(last_error))


# -----------------------------
# Workflow stages
# -----------------------------
def stage_planning(client, topic, level, duration, hours, goals):
    system = """You are the Planning Agent of an AI Study Pack Generator.

Create a practical, personalized study plan. Do not invent academic sources.
Return concise Markdown containing:
1. Learning objectives
2. Prerequisites
3. Weekly schedule
4. Key concepts
5. Recommended study sequence

Keep the response under 500 words."""

    prompt = f"""Topic: {topic}
Student level: {level}
Duration: {duration} weeks
Study time: {hours} hours/day
Student goals: {goals or "General mastery"}

Create a structured study plan."""

    return call_ai(
        client,
        system,
        prompt,
        temperature=0.4,
        max_tokens=700,
    )


def stage_content(client, topic, level, plan):
    system = """You are the Content Generation Agent.

Using the supplied study plan, create concise, useful study notes.
Explain concepts accurately at the requested level.

Include:
- Main concepts
- Important terms
- Short examples
- Common mistakes
- A revision checklist

Do not repeat the entire study plan.
Keep the response under 1,000 words."""

    prompt = f"""Topic: {topic}
Level: {level}

PLANNING CONTEXT:
{limit_text(plan, 1200)}

Generate the study content based on this plan."""

    return call_ai(
        client,
        system,
        prompt,
        temperature=0.4,
        max_tokens=1200,
    )


def stage_assessment(client, topic, level, content):
    system = """You are the Assessment Agent.

Create an assessment using only the supplied study content.

Include:
- 6 multiple-choice questions
- 3 short-answer questions
- 1 application question
- An answer key

Keep questions concise. Do not test material that is not covered by the content."""

    prompt = f"""Topic: {topic}
Level: {level}

STUDY CONTENT:
{limit_text(content, MAX_CONTENT_CHARS)}

Generate the assessment and answer key."""

    return call_ai(
        client,
        system,
        prompt,
        temperature=0.3,
        max_tokens=900,
    )


def stage_review(client, topic, plan, content, assessment):
    system = """You are a concise quality-review agent.

Review the study materials and return ONLY these sections:

## Strengths
## Issues Found
## Required Fixes
## Quality Score

Keep the whole review under 350 words.
Do not rewrite the study pack.
Do not explain hidden reasoning or your internal process."""

    prompt = f"""Topic: {topic}

PLAN:
{limit_text(plan, 1000)}

CONTENT:
{limit_text(content, 2200)}

ASSESSMENT:
{limit_text(assessment, 1600)}

Give a brief, direct quality review."""

    return call_ai(
        client,
        system,
        prompt,
        temperature=0.2,
        max_tokens=1200,
    )

def stage_refinement(client, topic, level, plan, content, assessment, review):
    system = """You are the Final Refinement Agent.

Create a polished final study pack from the supplied material and review.

Use exactly these Markdown headings:
# Study Roadmap
# Learning Objectives
# Study Notes
# Quick Revision
# Practice Questions
# Answer Key
# Final Review Tips

Use the quality review to fix important issues.
Do not mention internal workflow stages or that you are an AI.
Keep the final pack concise and useful."""

    prompt = f"""Topic: {topic}
Level: {level}

PLAN SUMMARY:
{limit_text(plan, 1000)}

STUDY CONTENT:
{limit_text(content, 3200)}

ASSESSMENT:
{limit_text(assessment, 1800)}

QUALITY REVIEW:
{limit_text(review, 1200)}

Create the final refined study pack."""

    return call_ai(
        client,
        system,
        prompt,
        temperature=0.3,
        max_tokens=1400,
    )


# -----------------------------
# Safe stage runner
# -----------------------------
def run_stage(name, fn, *args):
    try:
        with st.status(f"{name} in progress...", expanded=False):
            result = fn(*args)

        if not result or not str(result).strip():
            raise RuntimeError("The AI returned an empty response.")

        return result, None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# -----------------------------
# User interface
# -----------------------------
st.title("Ã°Å¸â€œÅ¡ AI Study Pack Generator")
st.caption(
    "Multi-stage AI workflow: Planning Ã¢â€ â€™ Content Ã¢â€ â€™ Assessment Ã¢â€ â€™ Review Ã¢â€ â€™ Refinement"
)

with st.sidebar:
    st.header("Student Profile")

    topic = st.text_input(
        "Study topic",
        placeholder="e.g., Machine Learning",
    )

    level = st.selectbox(
        "Level",
        ["Beginner", "Intermediate", "Advanced"],
    )

    duration = st.number_input(
        "Duration (weeks)",
        min_value=1,
        max_value=52,
        value=4,
    )

    hours = st.number_input(
        "Study hours/day",
        min_value=0.5,
        max_value=12.0,
        value=2.0,
        step=0.5,
    )

    goals = st.text_area(
        "Learning goals",
        placeholder="What do you want to achieve?",
    )

    generate = st.button(
        "Ã°Å¸Å¡â‚¬ Generate Study Pack",
        type="primary",
        use_container_width=True,
    )


if generate:
    if not topic.strip():
        st.error("Please enter a study topic.")
        st.stop()

    try:
        client = get_client()
    except Exception as exc:
        st.error(str(exc))
        st.info(
            "For Streamlit Cloud, add GROQ_API_KEY under Settings Ã¢â€ â€™ Secrets."
        )
        st.stop()

    # Clear only results from a previous generation.
    for key in ["plan", "content", "assessment", "review", "final"]:
        st.session_state.pop(key, None)

    # Stage 1
    plan, err = run_stage(
        "Stage 1 Ã¢â‚¬â€ Planning",
        stage_planning,
        client,
        topic,
        level,
        duration,
        hours,
        goals,
    )

    if err:
        st.error(f"Planning failed: {err}")
        st.stop()

    st.session_state["plan"] = plan

    # Stage 2
    content, err = run_stage(
        "Stage 2 Ã¢â‚¬â€ Content Generation",
        stage_content,
        client,
        topic,
        level,
        plan,
    )

    if err:
        st.error(f"Content generation failed: {err}")
        st.stop()

    st.session_state["content"] = content

    # Stage 3
    assessment, err = run_stage(
        "Stage 3 Ã¢â‚¬â€ Assessment",
        stage_assessment,
        client,
        topic,
        level,
        content,
    )

    if err:
        st.error(f"Assessment failed: {err}")
        st.stop()

    st.session_state["assessment"] = assessment

    # Stage 4
    review, err = run_stage(
        "Stage 4 Ã¢â‚¬â€ Review & Quality Check",
        stage_review,
        client,
        topic,
        plan,
        content,
        assessment,
    )

    if err:
        st.error(f"Review failed: {err}")
        st.stop()

    st.session_state["review"] = review

    # Stage 5
    final_pack, err = run_stage(
        "Stage 5 Ã¢â‚¬â€ Refinement",
        stage_refinement,
        client,
        topic,
        level,
        plan,
        content,
        assessment,
        review,
    )

    if err:
        st.error(f"Refinement failed: {err}")
        st.stop()

    st.session_state["final"] = final_pack
    st.success("Study pack generated successfully.")


# -----------------------------
# Results
# -----------------------------
if "final" in st.session_state:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Final Pack", "Planning", "Content", "Assessment", "Review"]
    )

    with tab1:
        st.markdown(st.session_state["final"])
        st.download_button(
            "Ã¢Â¬â€¡Ã¯Â¸Â Download Study Pack",
            data=st.session_state["final"],
            file_name="ai_study_pack.md",
            mime="text/markdown",
        )

    with tab2:
        st.markdown(st.session_state["plan"])

    with tab3:
        st.markdown(st.session_state["content"])

    with tab4:
        st.markdown(st.session_state["assessment"])

    with tab5:
        st.markdown(st.session_state["review"])

else:
    st.info(
        "Enter your study details in the sidebar and click Generate Study Pack."
    )
