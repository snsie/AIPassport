import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_timeline import timeline
from openai import OpenAI
import json
import aipassport_config as cfg

header_cols = st.columns(3)
with header_cols[1]:
    st.image("assets/images/headers/1.1_header.png", width=300)

st.markdown(
    """
Artificial intelligence (AI) can seem mysterious and complex, but at its core AI is a tool built by humans
to solve specific problems. This subsection demystifies AI in two ways.

First you will place today's capabilities in **historical context** with an interactive AI timeline, then
test your own assumptions in **AI: Fact or Fiction?**, which gives you immediate, evidence-based feedback.

Second, you will follow a **deployed clinical model through its lifecycle**. Watch what happens to its
accuracy when the patient population shifts, and practice data validation to solve the problem before an
incorrect prediction reaches a clinician.
"""
)

# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — Interactive AI Timeline
# ═══════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    st.markdown("## :material/touch_app: Interactive AI Timeline")

    st.markdown(
        """
    **Artificial Intelligence (AI)** has evolved from a bold concept to a transformative reality
    reshaping science, medicine, industry, and everyday life. This interactive timeline explores key
    milestones in the history of AI, from the theoretical groundwork laid by Alan Turing in the 1950s
    to the explosive rise of generative models and multimodal agents in the 2020s.

    As you scroll through the timeline, consider how each breakthrough reflects the state of computing
    at the time, and how it contributes to a larger story of increasing intelligence, autonomy, and impact.
"""
    )

    with open("assets/widgets/1.1_ai_timeline.json", "r") as f:
        timeline_data = f.read()

    with st.container(border=True):
        timeline(timeline_data, height=600)


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — AI: Fact or Fiction?
# ═══════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    st.markdown("## :material/gavel: AI: Fact or Fiction?")
    st.caption(
        "**Note:** The following activity uses generative AI to automatically provide feedback. Accuracy and appropriateness of responses is not guaranteed."
    )

    st.markdown(
        """
    Artificial Intelligence can feel like a mysterious black box — surrounded by hype, myths, and
    sometimes even fear. Some statements about AI reflect real technical capabilities and limitations;
    others are based on outdated ideas or science fiction. As AI grows more powerful and more visible in
    clinical practice, telling fact from fiction becomes a professional skill.

    Enter any statement you have heard or believed about AI — technical or philosophical — and the
    built-in assistant will help you evaluate whether it is accurate, misleading, or just plain wrong.
    """
    )

    # LLM configuration
    model_id = cfg.DEFAULT_MODEL
    navigator_api_key = st.secrets.get("NAVIGATOR_TOOLKIT_API_KEY")

    with open("assets/llm/1.1_gemini_system_instruction.txt", "r") as f:
        system_instruction = f.read()

    with open("assets/llm/1.1_gemini_response_schema.json", "r") as f:
        response_schema = json.load(f)

    if "m1_fof_statement" not in st.session_state:
        st.session_state.m1_fof_statement = ""
    if "m1_fof_verdict" not in st.session_state:
        st.session_state.m1_fof_verdict = ""

    def submit_statement():
        """on_change callback: clearing the widget key here is only legal in a callback."""
        if not client:
            return

        st.session_state.m1_fof_statement = st.session_state.m1_fof_input
        st.session_state.m1_fof_input = ""

        with llm_container:
            with st.spinner("Thinking...", show_time=True):
                # Use OpenAI SDK for NaviGator Toolkit
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": st.session_state.m1_fof_statement},
                    ],
                    response_format={"type": "json_object"},
                )

        st.session_state.m1_fof_verdict = json.loads(response.choices[0].message.content)

        # ── Share with AI Guide ──
        V = st.session_state.m1_fof_verdict
        st.session_state["_live_state"] = (
            f"User evaluated the statement: '{st.session_state.m1_fof_statement}'. "
            f"Verdict: {V.get('verdict', 'Unknown')}. "
            f"Explanation Summary: {V.get('verdict_explanation', {}).get('verdict_explanation_summary', 'N/A')}"
        )

    client = (
        OpenAI(api_key=navigator_api_key, base_url=cfg.NAVIGATOR_TOOLKIT_BASE_URL)
        if navigator_api_key
        else None
    )

    llm_container = st.container(border=True)
    with llm_container:
        if not navigator_api_key:
            st.info("AI feedback is unavailable because NAVIGATOR_TOOLKIT_API_KEY is not configured.")

        st.text_input(
            "Enter any statement about AI you'd like to evaluate.",
            placeholder="e.g., AI can think like a human.",
            key="m1_fof_input",
            on_change=submit_statement if client else None,
            disabled=not client,
        )

        if st.session_state.m1_fof_verdict != "":
            V = st.session_state.m1_fof_verdict
            explanation_block = V["verdict_explanation"]

            verdict = V["verdict"]
            explanation = explanation_block["verdict_explanation_summary"]
            limitations = explanation_block["limitations"]
            challenges = explanation_block["challenges"]
            requirements = explanation_block["future_requirements"]

            real_world_examples = V["real_world_examples"]
            research_examples = V["research_papers"]
            ml_concepts = V["high_level_machine_learning_concepts"]
            datasets = V["relevant_public_datasets"]
            research_directions = V["potential_research_directions"]

            st.markdown(f'Is "**{st.session_state.m1_fof_statement}**" fact or fiction?')

            if verdict in ["FACT", "MOSTLY FACT", "CURRENTLY FACT"]:
                icon, fn = ":material/thumb_up:", st.success
            elif verdict in ["FICTION", "MOSTLY FICTION", "CURRENTLY FICTION"]:
                icon, fn = ":material/thumb_down:", st.error
            elif verdict in ["MISLEADING", "NOT A STATEMENT", "MALICIOUS"]:
                icon, fn = ":material/report:", st.warning
            else:
                icon, fn = ":material/question_mark:", st.info

            ai_concept_str = ", ".join(ml_concepts)

            fn(
                f"""
                **Statement:** {st.session_state.m1_fof_statement} \n\n
                **Related AI concepts:** {ai_concept_str} \n\n
                # {icon} {verdict}
                {explanation}
            """
            )

            st.info("Click the panels below for more information about your statement.")

            with st.expander("Real-World Biomedical Applications", icon=":material/build:"):
                for i, example in enumerate(real_world_examples):
                    st.markdown(f"{i+1}. {example}")

            with st.expander(
                "Limitations, Challenges, and Future Requirements", icon=":material/build:"
            ):
                st.markdown(limitations)
                st.markdown(challenges)
                st.markdown(requirements)

            with st.expander("Research Opportunities", icon=":material/build:"):
                cols_ro = st.columns(len(research_directions), border=False)
                for i, example in enumerate(research_directions):
                    with cols_ro[i]:
                        st.markdown(f"{example}")

            with st.expander("AI Concepts", icon=":material/build:"):
                cols_ac = st.columns(len(ml_concepts), border=False)
                for i, example in enumerate(ml_concepts):
                    with cols_ac[i]:
                        st.markdown(f"**{example}**")

            with st.expander("Datasets", icon=":material/build:"):
                cols_d = st.columns(len(datasets), border=False)
                for i, example in enumerate(datasets):
                    with cols_d[i]:
                        st.markdown(f"{example}")

            with st.expander("Further Reading", icon=":material/build:"):
                for i, example in enumerate(research_examples):
                    st.markdown(f"{i+1}. {example}")


# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — The AI Lifecycle: a deployed model under data drift
# ═══════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    st.markdown("## :material/timeline: The AI Lifecycle in Deployment")

    st.markdown(
        """
    You are a cardiologist at a hospital that deployed an AI model two years ago to flag patients at high
    risk of a heart disease event during routine visits. The model performed well at launch. But patient
    populations shift, and now one of your nurses mentions the model's risk flags feel "off".

    An AI project does not end when the model is trained. It is deployed, it is monitored, the patient
    population changes, and someone has to decide when to retrain. Below is a **simulated heart-disease
    risk predictor** running on simulated EHR (electronic health record) data: three model versions and
    three incoming data batches, each of 100 patients with `age`, `systolic_bp`, `cholesterol`, `bmi`,
    `smoker`, and the observed `outcome` (1 = heart disease event).

    **Deployment v3 contains data drift** — the population is older, heavier, and more of them smoke, and
    the relationship between the risk factors and the outcome has changed.
    """
    )

    def _accuracy(y_true, y_pred):
        return (np.array(y_true) == np.array(y_pred)).mean()

    def _confusion_matrix(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn_ = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return np.array([[tn, fp], [fn_, tp]])

    def _roc_auc(y_true, y_prob):
        y_true, y_prob = np.array(y_true), np.array(y_prob)
        if len(np.unique(y_true)) != 2:
            return np.nan
        pos, neg = y_prob[y_true == 1], y_prob[y_true == 0]
        return (pos[:, None] > neg).mean() + 0.5 * (pos[:, None] == neg).mean()

    @st.cache_data
    def make_deployment_batches():
        rng = np.random.default_rng(2024)

        def make_patients(n, drift=False):
            age = rng.integers(40, 82, n)
            systolic_bp = rng.normal(132, 17, n) + (8 if drift else 0)
            cholesterol = rng.normal(222, 52, n) + (15 if drift else 0)
            bmi = rng.normal(28, 7, n) + (2 if drift else 0)
            smoker = rng.binomial(1, 0.37 if drift else 0.32, n)

            if drift:
                coef, intercept = np.array([0.03, 0.04, 0.018, 0.04, 0.2]), -13.0
            else:
                coef, intercept = np.array([0.04, 0.025, 0.012, 0.08, 0.7]), -11.7

            features = np.column_stack([age, systolic_bp, cholesterol, bmi, smoker])
            prob = 1 / (1 + np.exp(-(features @ coef + intercept)))

            return pd.DataFrame(
                {
                    "age": age,
                    "systolic_bp": systolic_bp.round(),
                    "cholesterol": cholesterol.round(),
                    "bmi": bmi.round(1),
                    "smoker": smoker,
                    "outcome": rng.binomial(1, prob),
                }
            )

        return [
            make_patients(100, drift=False),  # Deployment v1
            make_patients(100, drift=False),  # Deployment v2 (pre-drift)
            make_patients(100, drift=True),   # Deployment v3 (drifted)
        ]

    batches = make_deployment_batches()
    batch_names = ["Deployment v1 (Initial)", "Deployment v2 (Stable)", "Deployment v3 (Data Drift)"]

    def deployed_model_predict(X, version=1):
        if version == 2:
            coef, intercept = np.array([0.041, 0.026, 0.013, 0.076, 0.6]), -11.5
        elif version == 3:
            coef, intercept = np.array([0.03, 0.04, 0.018, 0.04, 0.2]), -13.0
        else:
            coef, intercept = np.array([0.04, 0.025, 0.012, 0.08, 0.7]), -11.7
        xb = (X[["age", "systolic_bp", "cholesterol", "bmi", "smoker"]] @ coef) + intercept
        return 1 / (1 + np.exp(-xb))

    st.markdown("**Sample of the incoming data:**")
    st.dataframe(batches[0].head(), width="stretch")

    st.markdown("#### Pick a model version and the batch to monitor it on")

    sim_cols = st.columns(2)
    with sim_cols[0]:
        version = st.selectbox(
            "Deployed model version:",
            options=[
                "Model v1 (trained on Deployment v1)",
                "Model v2 (retrained on Deployment v2)",
                "Model v3 (retrained on Deployment v3)",
            ],
            key="m1_lc_model_version",
        )
    with sim_cols[1]:
        batch_label = st.selectbox(
            "New incoming data batch:",
            options=[f"{i+1}: {name}" for i, name in enumerate(batch_names)],
            key="m1_lc_batch",
        )

    model_ver = int(version[-2])
    batch_ver = int(batch_label.split(":")[0]) - 1
    X_test = batches[batch_ver]
    y_test = X_test["outcome"]

    y_prob = deployed_model_predict(X_test, version=model_ver)
    y_pred = (y_prob >= 0.5).astype(int)

    st.write(f"Evaluating **{version}** on **{batch_names[batch_ver]}**:")

    auc = _roc_auc(y_test, y_prob)
    metric_cols = st.columns(2)
    metric_cols[0].metric("Accuracy", f"{_accuracy(y_test, y_pred):.2f}")
    metric_cols[1].metric("ROC-AUC", "N/A" if np.isnan(auc) else f"{auc:.2f}")

    cm = _confusion_matrix(y_test, y_pred)
    chart_cols = st.columns(2)
    with chart_cols[0]:
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["No event", "Event"],
                y=["No event", "Event"],
                colorscale="Blues",
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 18},
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="True label",
            height=380,
            margin=dict(l=40, r=20, t=55, b=45),
        )
        st.plotly_chart(fig, width="stretch")
    with chart_cols[1]:
        fig2 = px.histogram(
            pd.DataFrame({"Predicted heart disease probability": y_prob}),
            x="Predicted heart disease probability",
            nbins=20,
            range_x=[0, 1],
            title="Distribution of predicted probabilities",
        )
        fig2.update_layout(
            yaxis_title="Patient count",
            height=380,
            margin=dict(l=40, r=20, t=55, b=45),
            bargap=0.08,
        )
        st.plotly_chart(fig2, width="stretch")

    with st.expander("View chart data as text (accessible alternative)"):
        st.markdown("**Confusion matrix** — rows are the true label, columns the prediction.")
        st.dataframe(
            pd.DataFrame(cm, index=["True: no event", "True: event"], columns=["Predicted: no event", "Predicted: event"]),
            width="stretch",
        )
        st.markdown("**Predicted probabilities** — how many patients fall in each 0.1-wide band.")
        prob_bands = pd.cut(y_prob, bins=np.arange(0, 1.1, 0.1), include_lowest=True)
        st.dataframe(
            prob_bands.value_counts().sort_index().rename("Patient count").to_frame(),
            width="stretch",
        )

    cfg.try_this(
        "run **Model v1** on each batch in turn. On which batch does its performance first drop "
        "noticeably? Then run **Model v3** on Deployment v3 — does retraining recover what was lost?"
    )
    st.text_area("Your notes on model drift and retraining:", key="m1_lc_drift_notes")

    st.markdown("#### Data validation: catching bad records before inference")
    st.markdown(
        """
    An AI system must check data integrity *before* it predicts or retrains. Below, ten random records
    from the batch you selected are checked against acceptable ranges. A record silently accepted here
    becomes a prediction a clinician may act on.
    """
    )
    cfg.try_this(
        "move the sliders and watch how the definition of \"valid\" changes what gets flagged."
    )

    row_samp = batches[batch_ver].sample(10, random_state=111).copy()
    val_cols = st.columns(2)
    with val_cols[0]:
        a_min, a_max = st.slider("Acceptable age range (years):", 40, 100, (45, 85), key="m1_lc_age")
        bp_min, bp_max = st.slider("Systolic BP (mmHg):", 90, 220, (90, 180), key="m1_lc_bp")
    with val_cols[1]:
        chol_min, chol_max = st.slider("Cholesterol (mg/dL):", 100, 350, (120, 340), key="m1_lc_chol")
        bmi_min, bmi_max = st.slider("BMI:", 10, 50, (15, 45), key="m1_lc_bmi")

    out_of_range = (
        ~row_samp["age"].between(a_min, a_max)
        | ~row_samp["systolic_bp"].between(bp_min, bp_max)
        | ~row_samp["cholesterol"].between(chol_min, chol_max)
        | ~row_samp["bmi"].between(bmi_min, bmi_max)
    )
    row_samp["Validation Flag"] = np.where(out_of_range, "🚨 Problem", "OK")
    st.dataframe(row_samp, width="stretch")
    st.caption(f"{int(out_of_range.sum())} of 10 sampled records would be held back for review.")

st.markdown(
    """
---
**Key takeaways**

- AI is a tool built by people to solve a stated problem. Every capability in the timeline arrived
  because someone framed a problem narrowly enough to make progress on it.
- A confident claim about AI is not evidence. Check what the system can actually do before you repeat it.
- **A model does not fail loudly.** Accuracy decays as the population drifts away from the training data,
  while the model keeps returning a number with the same confidence it always had.
- Retraining is a decision someone has to make, on evidence. Monitoring is what produces that evidence.
- **Validate the data before the model sees it.** A record accepted without checks becomes a prediction a
  clinician may act on.

**Resources**
- [eICU Collaborative Research Database](https://eicu-crd.mit.edu/) · [MIMIC-IV](https://physionet.org/content/mimiciv/)
- [MLflow](https://mlflow.org/) and [DVC](https://dvc.org/) for versioning models and data
"""
)
