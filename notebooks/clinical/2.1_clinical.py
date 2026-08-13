import streamlit as st

st.markdown(
    """
Some questions about an AI system cannot be settled by a number. Someone has to reason them through
first, and decide what would even count as harm. This subsection asks three of those questions:

1. **Which ethical principles are in tension** when a clinical AI system is deployed?
2. **How does a training-population mismatch become deployed harm?**
3. **When must a human stay in the loop** — and what do you do when the tool is already in use?

Subsection 2.2 then makes each of these measurable.
"""
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — The four principles in tension
# ═══════════════════════════════════════════════════════════════════════════
st.header("1. The Four Principles of Bioethics")

with st.expander("Read the case (click to expand)", expanded=True):
    st.markdown(
        """
    A hospital is piloting an AI system to predict disease risks and support early diagnosis, hoping to
    improve patient outcomes. The system uses large volumes of **de-identified patient data** —
    demographics, clinical histories, and lifestyle information. De-identified means no names or
    addresses, but the AI still analyzes broad health trends.

    **The dilemma:** for maximum accuracy, the model benefits from detailed geographic and demographic
    information. Those same details can permit **re-identification** — working out who an individual is,
    especially in unusual combinations of attributes or for rare diseases.

    If a patient's health information were re-identified and reached third parties — employers, insurers,
    cybercriminals — the consequences include discrimination, financial harm, and irreversible loss of
    privacy.
    """
    )

st.subheader("1.1 Which principles apply here?")
st.multiselect(
    "Select all that clearly apply in this scenario:",
    [
        "Autonomy (respecting patient choice and privacy)",
        "Beneficence (acting to benefit the patient and population)",
        "Non-maleficence (do no harm)",
        "Justice (fairness and equity in healthcare)",
    ],
    key="m2_ethics_principles",
)

if st.button("Show example — principles", key="m2_ethics_principles_btn"):
    st.success(
        "All four principles are relevant:\n"
        "- **Autonomy**: Patients expect control over their private information; re-identification risks violate their autonomy and privacy.\n"
        "- **Beneficence**: The AI could improve diagnosis and outcomes (population benefit).\n"
        "- **Non-maleficence**: Re-identification could cause real harm (discrimination, financial harm).\n"
        "- **Justice**: If certain groups are more at risk of re-identification (rare conditions, small communities), or are excluded in order to protect privacy, this raises fairness concerns."
    )

st.subheader("1.2 Which principles are in conflict, and how?")
st.text_area(
    "Name the specific tension — not the principles in the abstract, but what one costs the other here:",
    height=140,
    key="m2_ethics_conflict",
)

if st.button("Show example — conflicts", key="m2_ethics_conflict_btn"):
    st.info(
        "- **Beneficence** (improving care via a better model) vs. **Autonomy** and **Non-maleficence** "
        "(protecting privacy, preventing harm): the more detailed the data, the more the AI helps "
        "patients — and the higher the risk of re-identification and harm.\n"
        "- **Justice** also conflicts if privacy risks are unevenly distributed, or if some populations "
        "are excluded from the data specifically in order to protect them."
    )

st.subheader("1.3 Which principle should take precedence? Why?")
st.text_area(
    "Defend your view: which principle should guide clinicians and hospital policy here, and why?",
    height=140,
    key="m2_ethics_precedence",
)

if st.button("Show example — precedence", key="m2_ethics_precedence_btn"):
    st.info(
        "Example: while beneficence is important, **non-maleficence** (do no harm) and **autonomy** "
        "(patient privacy) should take precedence — especially where a privacy breach causes irreversible "
        "harm. The hospital must put safeguards in place so that no patient can be re-identified, even if "
        "that reduces model accuracy somewhat; otherwise trust is lost and harm may follow."
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — Where bias enters
# ═══════════════════════════════════════════════════════════════════════════
st.header("2. How a Population Mismatch Becomes Deployed Harm")

with st.expander("Read the case (click to expand)", expanded=True):
    st.markdown(
        """
    A large hospital is the only level-one trauma center within a 100-mile radius of a small city in the
    Southern United States. Because of this it receives an unusual concentration of traumatic
    injuries — vehicle crashes, shootings, catastrophic injuries — through its emergency room.

    Staff and administration want a better way to determine which patients should receive the most
    immediate care. The proposal is an algorithm that **ranks patients by acuity**, calculated from the
    symptoms and demographic data entered by hospital staff.

    Answer three things:

    - What are the possible **vectors of bias** that might affect patient care?
    - What should the hospital consider **before** deploying the tool?
    - What are the possible negative outcomes?
    """
    )

st.text_area(
    "Your answer — address the data, the bias vectors, and the practical consequences:",
    height=180,
    key="m2_bias_response",
)

if st.button("Reveal example and guidance", key="m2_bias_example_btn"):
    st.success(
        """
**Example:**

The vectors of bias for the algorithm include the data and the people who will interpret the guidance
from the algorithm. As the only trauma center in a 100-mile radius, the hospital will receive all kinds of
terrible injuries that other hospitals may not receive, therefore, if the data that was used to train the
algorithm is not similar, the resulting guidance will be off-base. Further, although the algorithms will
only be used to offer guidance, some clinicians will think that the algorithm cannot be wrong, and they
won't critically consider the results.

**Guidance:**
- Consider how the unique patient population (regional, demographic, trauma-specific) may or may not be
  reflected in the data used to train the model.
- Consider the risk that social, demographic, or subjective inputs introduce or amplify bias.
- Reflect on the consequences: inequities in care, over-reliance on the algorithm, errors propagating
  because nobody expected the tool to be wrong.
"""
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — Human oversight
# ═══════════════════════════════════════════════════════════════════════════
st.header("3. When a Human Must Stay in the Loop")

with st.expander("Read the case (click to expand)", expanded=True):
    st.markdown(
        """
    A large hospital has implemented an **AI transcription system** for staff to use in care settings. The
    intent was more accurate notes in patient records. After about a month of use across various care
    settings, staff reviewed the resulting transcripts and found that although the tool transcribed
    patient interviews, it also:

    - **made up segments of conversations that did not happen**;
    - made certain patients appear to be **behaving aggressively** with staff, when that behaviour was not
      present; and
    - was **markedly less accurate** in conversations with patients with accents, from around the US or
      otherwise.

    **What are the possible routes the hospital could take after reviewing the transcription data? What
    should the hospital do? Explain your answer.**
    """
    )

st.markdown(
    """
This is the harder version of the oversight question. The tool is already deployed, notes are already in
charts, and every option costs something. Say what you would do, and be explicit about what your choice
gives up.
"""
)

st.text_area(
    "Your response:",
    height=200,
    key="m2_oversight_response",
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Closing reflection
# ═══════════════════════════════════════════════════════════════════════════
st.header("Consider this:")

st.markdown(
    """
Each case above was resolved by reasoning. None of it is verifiable yet — which is the problem subsection
2.2 exists to solve.

- How would you **monitor** fairness *after* deployment, rather than argue about it beforehand?
- Which of the three cases could have been caught by a metric, and which needed a person to notice?
"""
)
st.text_area("Your response:", height=120, key="m2_ethics_reflection")

st.markdown(
    """
---
**Key takeaways**

- **Autonomy** = respecting patients' wishes and privacy · **Beneficence** = doing good for the
  patient and population · **Non-maleficence** = avoiding harm · **Justice** = fairness in the
  distribution of risks and benefits.
- **The four principles conflict in practice.** Naming which one you let win, and what that cost the
  others, is the whole of the reasoning — listing all four is not.
- No algorithm is objective or immune to bias. Both the technical design and the human interpretation can
  perpetuate or reduce inequity.
- **A population mismatch becomes harm through people, not just data.** A model trained on the wrong
  population and trusted by a clinician who assumes it cannot be wrong fails twice over.
- A tool that is *usually* right in a high-stakes setting creates a new failure mode: nobody checks it.
- **Some harms have no metric yet.** Reasoning is what tells you which metric to go build in 2.2.

**Further reading:** [Theory and Bioethics — the four principles (Stanford)](https://plato.stanford.edu/entries/theory-bioethics/) ·
[AMA: Advancing health care AI through ethics, evidence and equity](https://www.ama-assn.org/practice-management/digital-health/advancing-health-care-ai-through-ethics-evidence-and-equity) ·
[WHO: Ethics & Governance of AI for Health](https://www.who.int/publications/i/item/9789240029200)
"""
)
