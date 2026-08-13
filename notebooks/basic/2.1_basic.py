import streamlit as st

st.markdown(
    """
Some questions about an AI system cannot be settled by a number. Someone has to reason them through
first, and decide what would even count as harm. This subsection asks three of those questions:

1. **Which ethical principles are in tension** when a biomedical AI system is deployed?
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
    **Dr. Lee**, a biomedical researcher, has developed an AI model that predicts disease risk from
    genetic data. The model can identify high-risk individuals early, enabling preventive care.

    **However:** during testing it becomes clear that the algorithm disproportionately predicts higher risk
    for certain racial and ethnic groups, because of biases in the training data. Deploying it could benefit
    many patients while also reinforcing existing healthcare disparities.

    **Dr. Lee must decide** whether to launch the model while working to improve its fairness, or to delay
    deployment entirely in order to refine the algorithm.
    """
    )

st.subheader("1.1 Which principles apply here?")
st.multiselect(
    "Select all that clearly apply in this scenario:",
    [
        "Autonomy (respect for persons)",
        "Beneficence (do good)",
        "Non-maleficence (do no harm)",
        "Justice (fair and equitable treatment)",
    ],
    key="m2_ethics_principles",
)

if st.button("Show example — principles", key="m2_ethics_principles_btn"):
    st.success(
        "All four principles are relevant:\n"
        "- **Autonomy**: Participants have a right to be informed and to choose how their genetic risk data are used.\n"
        "- **Beneficence**: The model could benefit many people by enabling early preventive care.\n"
        "- **Non-maleficence**: There is real potential for harm through faulty risk predictions and reinforced disparities.\n"
        "- **Justice**: Model bias could increase health inequity for specific racial and ethnic groups."
    )

st.subheader("1.2 Which principles are in conflict, and how?")
st.text_area(
    "Name the specific tension — not the principles in the abstract, but what one costs the other here:",
    height=140,
    key="m2_ethics_conflict",
)

if st.button("Show example — conflicts", key="m2_ethics_conflict_btn"):
    st.info(
        "- **Beneficence** (do good for many) is in tension with **Justice** (fairness) and "
        "**Non-maleficence** (avoid harm): deploying the model may help most people while harming or "
        "unfairly treating minority groups.\n"
        "- **Autonomy** enters as soon as you ask whether the people in the training data agreed to this "
        "use of their genetic information."
    )

st.subheader("1.3 Which principle should take precedence? Why?")
st.text_area(
    "Which bioethical principle should guide the decision here? Justify your answer:",
    height=140,
    key="m2_ethics_precedence",
)

if st.button("Show example — precedence", key="m2_ethics_precedence_btn"):
    st.info(
        "Example: *Justice* should take precedence. If the model increases disparities or harms "
        "disadvantaged groups, it undermines the goals of medicine. A just system must ensure that AI "
        "benefits are equitable, even when that requires delay and extra work."
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — Where bias enters
# ═══════════════════════════════════════════════════════════════════════════
st.header("2. How a Population Mismatch Becomes Deployed Harm")

with st.expander("Read the case (click to expand)", expanded=True):
    st.markdown(
        """
    A large organization wants to build an algorithm that helps major cities combat public health crises.
    It will use real-time monitoring of traditional and digital media, plus search trends, to identify
    possible emerging disease clusters.

    In building the algorithm, the organization uses several training datasets, including data from areas
    where:

    - the **population demographics** do **not** reflect those of most major cities; and
    - the **income demographics** do **not** reflect those of most major cities.

    Answer three things:

    - What are the possible **vectors of bias** that might affect the accuracy of the algorithm?
    - What should the organization consider **before** marketing the tool?
    > - What are the possible negative outcomes?
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

The vectors of bias for the algorithm include using data that does not reflect the kinds of people in the
cities that will adopt the system, and using income as a factor. Using this kind of data will produce
guidance that is off-base. Further, although the algorithm will only be used to offer guidance, some public
health officials will think that the algorithm cannot be wrong, and they won't critically consider the
results.

**Guidance:**
- Think about whether the data "matches" the population it is meant to serve.
- Consider possible misrepresentations, especially where demographic and income factors differ between
  training and deployment.
- Reflect on the risks if policymakers rely too heavily or too uncritically on algorithmic results.
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
    In researching the effectiveness of public health campaigns, a team has decided to conduct qualitative
    interviews and wants to use an **AI transcription system** to produce more accurate notes. After
    testing the tool in various recording settings, study staff reviewed the transcripts and found that
    although the tool transcribed the interviews, it also:

    - **made up segments of conversations that did not happen**;
    - made certain interviewees appear to be **behaving aggressively** with the interviewers, when that
      behaviour was not present;
    - was **markedly less accurate** with interviewees with accents, from around the US or otherwise; and
    - **recorded personally identifiable information in the metadata** of the transcripts.

    **Should the research team use the tool given these findings? Explain your answer, weighing the
    benefits and drawbacks of going forward.**
    """
    )

st.markdown(
    """
This is the harder version of the oversight question. The tool has already touched the data, the interviews
cannot be re-run cheaply, and every option costs something. Say what you would do, and be explicit about
what your choice gives up.
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

- **Autonomy** = respecting participants' wishes and privacy · **Beneficence** = promoting well-being ·
  **Non-maleficence** = avoiding harm · **Justice** = treating all fairly and addressing disparities.
- **The four principles conflict in practice.** Naming which one you let win, and what that cost the
  others, is the whole of the reasoning — listing all four is not.
- No dataset is perfectly representative, and no algorithm is immune to bias. Both the data *and* the human
  interpretation shape the impact.
- **A population mismatch becomes harm through people, not just data.** A model trained on the wrong
  population and trusted by someone who assumes it cannot be wrong fails twice over.
- A tool that is *usually* right in a high-stakes setting creates a new failure mode: nobody checks it.
- **Some harms have no metric yet.** Reasoning is what tells you which metric to go build in 2.2.

**Further reading:** [Theory and Bioethics — the four principles (Stanford)](https://plato.stanford.edu/entries/theory-bioethics/) ·
[Fairness and Machine Learning (Barocas, Hardt & Narayanan)](https://fairmlbook.org/) ·
[WHO: Ethics & Governance of AI for Health](https://www.who.int/publications/i/item/9789240029200)
"""
)
