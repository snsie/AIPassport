import streamlit as st
import time
from openai import OpenAI
import aipassport_config as cfg

header_cols = st.columns(3)
with header_cols[1]:
    st.image("assets/images/headers/7.1_header.png")

st.markdown(
    """
Biomedical researchers are increasingly looking to incorporate artificial intelligence into their research
programs. But designing a strong AI-enabled experiment requires more than selecting a model — it means
establishing clear hypotheses, understanding data limitations, choosing appropriate techniques, and planning
for baseline and comparative evaluations.

This subsection walks that path in three steps.

1. **Find the opening** — name a gap in your own field and see what AI approach could address it.
2. **Design the study** — describe the experiment and get structured feedback on its feasibility and clarity.
3. **Make it rigorous** — generate the datasheet or model card that documents what you built.

Subsection 7.2 then takes this design out into the world: to funders, to lay audiences, to a critic, and to a
research-integrity case.
"""
)

st.caption(
    "**Note:** These activities use a generative AI model to provide suggestions and feedback. This is an "
    "educational tool, not a peer review system."
)

# ═══════════════════════════════════════════════════════════════════════════
# Shared LLM helper — defined once, used by every activity on this page.
# ═══════════════════════════════════════════════════════════════════════════

# Module 7's activities have always run on this model rather than cfg.DEFAULT_MODEL.
# Kept as-is so the prompts behave exactly as they were written and tested.
MODEL_ID = "gemma-3-27b-it"

MARKDOWN_PREAMBLE = (
    "Respond in clear, readable markdown. Do NOT return JSON or code blocks.\n"
    "Use headers (###), bullet points, and bold text to organize your feedback.\n\n"
)

navigator_api_key = st.secrets.get("NAVIGATOR_TOOLKIT_API_KEY")
client = (
    OpenAI(api_key=navigator_api_key, base_url=cfg.NAVIGATOR_TOOLKIT_BASE_URL)
    if navigator_api_key
    else None
)

if not navigator_api_key:
    st.info(
        "AI feedback is unavailable because NAVIGATOR_TOOLKIT_API_KEY is not configured. You can still read "
        "each activity's brief and draft your response."
    )


@st.cache_data
def load_system_instruction(path, add_markdown_preamble=False):
    with open(path, "r") as f:
        instruction = f.read()
    return (MARKDOWN_PREAMBLE + instruction) if add_markdown_preamble else instruction


def run_llm_activity(
    slug,
    instruction_path,
    input_label,
    placeholder_text=None,
    add_markdown_preamble=False,
):
    """Render one LLM activity.

    Every widget key and session-state name is derived from `slug`, so several
    activities can live on one page without colliding. The submit path reproduces
    the pending-flag pattern the Module 7 notebooks arrived at: the button records
    the input and sets a flag, then st.rerun() lets the generation happen on a
    clean pass. Do not collapse these steps -- the flag is what makes
    button-triggered streaming fire reliably.
    """
    input_key = f"m7_{slug}_input"
    submitted_key = f"m7_{slug}_submitted"
    pending_key = f"m7_{slug}_pending"
    output_key = f"m7_{slug}_output"

    if submitted_key not in st.session_state:
        st.session_state[submitted_key] = ""
    if output_key not in st.session_state:
        st.session_state[output_key] = ""
    if pending_key not in st.session_state:
        st.session_state[pending_key] = False

    system_instruction = load_system_instruction(instruction_path, add_markdown_preamble)

    st.text_area(
        input_label,
        placeholder=placeholder_text,
        key=input_key,
        height=200,
        disabled=client is None,
    )

    if st.button(
        "✅ Submit",
        type="primary",
        width="stretch",
        key=f"m7_{slug}_submit",
        disabled=client is None,
    ):
        st.session_state[submitted_key] = st.session_state[input_key]
        st.session_state[pending_key] = True
        st.rerun()

    if st.session_state[pending_key]:
        st.session_state[pending_key] = False
        with st.container(border=True):
            output_placeholder = st.empty()
            try:
                with st.spinner("⏳ Generating response...", show_time=True):
                    response = client.chat.completions.create(
                        model=MODEL_ID,
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": st.session_state[submitted_key]},
                        ],
                    )
                full_response = response.choices[0].message.content
                curr = ""
                for line in full_response.split("\n"):
                    curr += line + "\n"
                    output_placeholder.markdown(curr + "▌")
                    time.sleep(0.04)
                output_placeholder.markdown(full_response)
                st.session_state[output_key] = full_response
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state[output_key] = ""
    elif st.session_state[output_key]:
        # Keep the last response on screen when a rerun comes from elsewhere on the page.
        with st.container(border=True):
            st.markdown(st.session_state[output_key])


# ═══════════════════════════════════════════════════════════════════════════
# Activities
# ═══════════════════════════════════════════════════════════════════════════

activity = st.radio(
    "Activity:",
    [
        "1. Find the opening",
        "2. Design the study",
        "3. Make it rigorous",
    ],
    horizontal=True,
    key="m7_design_activity",
)

st.divider()

if activity == "1. Find the opening":
    st.markdown("## :material/psychology: AI Solutions for Biomedical Challenges")
    st.markdown(
        """
    Start where you actually are. Name a specific challenge, limitation, or knowledge gap in your own field —
    the more concrete the better — and you will get a hypothetical AI approach that could address it.

    A gap stated as "we don't use AI yet" produces nothing useful. A gap stated as "we cannot predict which
    post-operative patients will deteriorate overnight, because the signal is spread across nursing notes and
    vitals nobody reads together" produces something you can build on.
    """
    )
    run_llm_activity(
        slug="gap",
        instruction_path="assets/llm/7.4_gemini_system_instruction.txt",
        input_label="Enter a challenge, limitation, or gap in knowledge in your field.",
        placeholder_text="e.g., We cannot identify which discharged heart-failure patients will be "
        "readmitted, because the risk factors are spread across notes, labs, and social context...",
    )

elif activity == "2. Design the study":
    st.markdown("## :material/psychology: Describe Your Biomedical AI Experiment Idea")
    st.markdown(
        """
    Now turn the opening into a study. Describe the experiment you would run: the question, the data, the
    approach, and how you would know whether it worked.

    Whether you have a specific project in mind or are still brainstorming, this returns structured feedback
    aimed at the feasibility and clarity of the design — which is where most AI study proposals actually fail.
    """
    )
    run_llm_activity(
        slug="experiment",
        instruction_path="assets/llm/7.1_gemini_system_instruction.txt",
        input_label="Describe your idea for a biomedical research experiment involving AI. "
        "**The more details, the better.**",
        placeholder_text="e.g., I want to use deep learning to predict postoperative complications "
        "based on EHR data and imaging...",
        add_markdown_preamble=True,
    )

else:
    st.markdown("## :material/psychology: Datasheet and Model Card Generator")
    st.markdown(
        """
    A design becomes rigorous when its limits are written down. Describe the dataset or model at the centre of
    your study and you will get a corresponding **datasheet** or **model card** — the document that states
    intended use, provenance, and known limitations.

    You built a smaller version of this in subsection 2.2. The point is the same and it is worth repeating:
    the artifact is what lets the next person inherit your caveats instead of rediscovering them.
    """
    )
    run_llm_activity(
        slug="datasheet",
        instruction_path="assets/llm/7.6_gemini_system_instruction.txt",
        input_label="Provide the name and/or description of a public biomedical dataset and/or AI model.",
        placeholder_text="e.g., MIMIC-IV, used to train a 30-day readmission classifier for heart "
        "failure patients at a single academic centre...",
    )

st.markdown(
    """
---
**Key takeaways**

- **A gap is a claim about the literature, not a hunch.** If you cannot say what has already been tried
  and why it fell short, you have a topic rather than an opening.
- A study design is judged on whether it could fail. Name the result that would tell you the hypothesis
  was wrong, or there is nothing to evaluate.
- **Choosing an AI method is the last decision, not the first.** It follows from the question, the data
  you can actually get, and the baseline you have to beat.
- A baseline is not optional. Without one, a good-looking number says nothing about whether the AI helped.
- **Documentation is part of the science.** A datasheet or model card is what lets the next person inherit
  your caveats instead of rediscovering them the hard way.
- Generative AI feedback is a rehearsal, not a review. Everything it returns still needs your judgement.

**Where this goes next**

The three artifacts above — a gap worth closing, a study design that survives scrutiny, and a datasheet that
states its limits — are the inputs to subsection 7.2. Keep them.
"""
)
