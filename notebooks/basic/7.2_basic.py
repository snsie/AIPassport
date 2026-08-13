import streamlit as st
import time
from openai import OpenAI
import aipassport_config as cfg

st.markdown(
    """
Subsection 7.1 produced a study design and a rigor artifact. A design nobody funds, nobody understands, and
nobody has stress-tested is not yet research.

Four activities, each aimed at a different audience — and the last one at the pressure that arrives when the
results are inconvenient.

1. **Funders** — turn the idea into an NIH-style project summary.
2. **Everyone else** — compress a paper into an elevator pitch a non-specialist actually follows.
3. **A critic** — get your idea attacked before a reviewer does it for you.
4. **Research integrity** — work through a misconduct case where the pressure is on you to stay quiet.
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
        "1. Write the proposal",
        "2. Pitch it plainly",
        "3. Invite the critique",
        "4. Handle the pressure",
    ],
    horizontal=True,
    key="m7_comm_activity",
)

st.divider()

if activity == "1. Write the proposal":
    st.markdown("## :material/psychology: Generate an NIH-Style Project Summary")
    st.markdown(
        """
    A project summary has to do a specific job in about thirty lines: say what problem you are solving, why it
    matters, what you will actually do, and why you are the one to do it. Describe your research idea and you
    will get a corresponding NIH-style summary to react against.

    Read what comes back critically. The useful question is not "is this good writing" but **"does this claim
    more than my design can deliver?"**
    """
    )
    run_llm_activity(
        slug="proposal",
        instruction_path="assets/llm/7.2_gemini_system_instruction.txt",
        input_label="Describe your research idea. **The more details, the better.**",
        placeholder_text="e.g., A prospective study of an AI early-warning score for post-operative "
        "deterioration, validated across three hospitals...",
    )

elif activity == "2. Pitch it plainly":
    st.markdown("## :material/psychology: Generate an Elevator Pitch")
    st.markdown(
        """
    Paste an abstract — or a full article — and get back an elevator-pitch version written for a broad
    audience.

    This is harder than it looks and it is not a writing exercise. Compressing a result forces you to decide
    what the finding actually *is*, and any hedge you drop on the way out is a hedge you have to be willing to
    defend dropping.
    """
    )
    run_llm_activity(
        slug="pitch",
        instruction_path="assets/llm/7.3_gemini_system_instruction.txt",
        input_label="Enter your research text (e.g., manuscript abstract, full article text, etc.)",
        placeholder_text="Paste an abstract here...",
    )

elif activity == "3. Invite the critique":
    st.markdown("## :material/psychology: Critique Generator for Biomedical AI Research Ideas")
    st.markdown(
        """
    Describe a biomedical AI research idea and get back a critique: the limitations a reviewer would raise, and
    suggestions for addressing them.

    Use the design you developed in 7.1. The objections you cannot answer yet are your actual to-do list — and
    it is considerably cheaper to find them here than in review.
    """
    )
    run_llm_activity(
        slug="critique",
        instruction_path="assets/llm/7.5_gemini_system_instruction.txt",
        input_label="Provide a biomedical AI research idea in as much detail as possible.",
        placeholder_text="e.g., A CNN trained on chest radiographs from one hospital to triage suspected "
        "pneumonia in the emergency department...",
    )

else:
    st.markdown("## :material/psychology: Navigating Research Misconduct")
    st.markdown(
        """
    Everything above assumed the results were honest and the incentives aligned. This activity assumes neither.

    Read the case, answer both questions in the box, and you will get an analysis of your reasoning plus
    proposed measures for preventing this kind of misconduct.
    """
    )

    with st.expander("**Read the case** (click to expand)", expanded=True):
        st.markdown(
            """
        A clinical trial was conducted by a well-known pharmaceutical company in collaboration with the
        University of Oxbridge to assess the efficacy of a new cancer drug. The results were published in a
        high-impact journal and showed positive outcomes, suggesting the drug significantly improved patient
        survival rates.

        **Dr. Smith**, a postdoctoral researcher in the university's medical department, was part of the
        research team responsible for collecting and analyzing patient data. During a routine review of the
        data files, Dr. Smith noticed irregularities — duplicated data points and altered timestamps. After
        further investigation, Dr. Smith found that a significant portion of the data had been **falsified** to
        show better outcomes than were observed.

        Dr. Smith reported these findings to the principal investigator, **Dr. Johnson**, who insisted the
        discrepancies were clerical errors and urged Dr. Smith to ignore them. Feeling pressured, Dr. Smith
        remained silent, and the data remained in the published study.
        """
        )

    st.markdown(
        """
    **Q1. What is the research misconduct in this case?**

    **Q2. Which of Dr. Johnson's and/or Dr. Smith's actions were unethical?**

    Answer both below, in one response. Q2 is the harder question — note that it asks about *both* people.
    """
    )

    run_llm_activity(
        slug="misconduct",
        instruction_path="assets/llm/7.7_gemini_system_instruction.txt",
        input_label="Provide your responses to Q1 and Q2:",
        placeholder_text="Q1. ...\n\nQ2. ...",
    )

st.markdown(
    """
---
**Key takeaways**

- A proposal that promises more than the design delivers is a problem you created for your future self.
- Compressing a finding for a lay audience forces you to decide what the finding is.
- An objection found in rehearsal is cheap. The same objection found in review is not.
- Research integrity is rarely a choice between obvious right and obvious wrong. It is usually a choice made
  under pressure, by someone junior, with a deadline.
"""
)
