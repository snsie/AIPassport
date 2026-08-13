import streamlit as st
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

Second, you will walk a molecular classification project through **the full AI lifecycle**. You make the
four decisions every such project demands, then read a single consolidated critique of the plan you built.
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
    research, telling fact from fiction becomes a professional skill.

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
# Part 3 — The AI Lifecycle: four decisions, one critique
# ═══════════════════════════════════════════════════════════════════════════
with st.container(border=True):
    st.markdown("## :material/timeline: The AI Project Lifecycle")

    st.markdown(
        """
    You are planning a **molecular classification project** — predicting whether compounds are active
    against a protein target.

    Every AI project makes the same four decisions, in the same order, and each one constrains the next.
    Make all four below, then submit to see how your plan holds up. *(No molecular data required — this is
    a walkthrough of the decisions, not the code.)*
    """
    )

    with st.form("m1_lifecycle_form"):
        st.markdown("**Stage 1 — Data collection.** Where will the labelled molecules come from?")
        data_source = st.radio(
            "Data source:",
            [
                "Public molecular database (e.g. ChEMBL, PubChem)",
                "In-house experimental dataset",
                "Simulated/generated molecules",
                "Not sure yet",
            ],
            key="m1_lc_source",
        )

        st.markdown("**Stage 2 — Preprocessing.** What has to happen before a model sees the data?")
        prepro_steps = st.multiselect(
            "Preprocessing steps (select all that apply):",
            [
                "Remove duplicate molecules",
                "Standardize chemical representations",
                "Calculate molecular descriptors/embeddings",
                "Handle missing values",
                "Scale/normalize numerical features",
                "Assign class labels",
                "None – ready to model",
            ],
            key="m1_lc_prepro",
        )

        st.markdown("**Stage 3 — Model selection.** What will you fit to those features?")
        model_choice = st.radio(
            "Modelling approach:",
            [
                "Logistic regression",
                "Random forest",
                "Neural network",
                "Support vector machine",
            ],
            key="m1_lc_model",
        )

        st.markdown("**Stage 4 — Validation.** How will you know whether it generalizes?")
        validation = st.multiselect(
            "Validation approach(es):",
            [
                "Simple train/test split",
                "Cross-validation",
                "External test set (different source)",
                "Leave-cluster-out validation (e.g., by scaffold or class)",
            ],
            key="m1_lc_validation",
        )

        submitted = st.form_submit_button("Review my lifecycle plan", type="primary")

    if submitted:
        notes = []

        # Stage 1
        if data_source == "Public molecular database (e.g. ChEMBL, PubChem)":
            notes.append(
                "**Data:** public databases give you scale, standardized structures, and rich metadata. "
                "Watch for reporting bias (actives are published more than inactives) and label-definition "
                "drift between assays — check that their label means what your question means."
            )
        elif data_source == "In-house experimental dataset":
            notes.append(
                "**Data:** in-house data matches your application closely, but is usually smaller, less "
                "standardized, and carries your lab's protocol as a hidden variable. Document the protocol "
                "and the missingness."
            )
        elif data_source == "Simulated/generated molecules":
            notes.append(
                "**Data:** simulated molecules are useful for augmentation and exploring novel chemistry, "
                "but they inherit the assumptions of the generator. Anything you conclude must be confirmed "
                "against real measurements."
            )
        else:
            notes.append(
                "**Data:** the source decision cannot be deferred — it determines your label quality, your "
                "biases, and what your model can honestly claim. Start by writing down what a positive label "
                "would have to mean."
            )

        # Stage 2
        if "None – ready to model" in prepro_steps or not prepro_steps:
            notes.append(
                "**Preprocessing:** raw SMILES strings are text — no model can consume them directly. At "
                "minimum you need descriptor or fingerprint calculation, deduplication, and a label column."
            )
        else:
            got = []
            if "Remove duplicate molecules" in prepro_steps:
                got.append("deduplication (prevents leakage between train and test)")
            if "Standardize chemical representations" in prepro_steps:
                got.append("standardization (tautomers and salts otherwise look like distinct molecules)")
            if "Calculate molecular descriptors/embeddings" in prepro_steps:
                got.append("featurization (the step that makes the data learnable at all)")
            if "Handle missing values" in prepro_steps:
                got.append("missing-value handling")
            if "Scale/normalize numerical features" in prepro_steps:
                got.append("scaling (matters for logistic regression and SVMs, not for trees)")
            if "Assign class labels" in prepro_steps:
                got.append("label assignment")
            notes.append("**Preprocessing:** you selected " + "; ".join(got) + ".")
            if "Calculate molecular descriptors/embeddings" not in prepro_steps:
                notes.append(
                    "⚠️ **Gap:** without descriptor or embedding calculation there are no features to "
                    "train on. This is the one step in Stage 2 that is not optional."
                )
            if "Remove duplicate molecules" not in prepro_steps:
                notes.append(
                    "⚠️ **Gap:** duplicate molecules that land on both sides of your split will inflate "
                    "every number you report in Stage 4."
                )

        # Stage 3
        model_notes = {
            "Logistic regression": "fast, interpretable, and a fair baseline for binary activity — but "
            "high-dimensional binary fingerprints need regularization, and it cannot represent interactions "
            "between substructures.",
            "Random forest": "handles high-dimensional sparse fingerprints well, resists overfitting, and "
            "gives you feature importances. A strong default for cheminformatics.",
            "Neural network": "can capture non-linear structure, especially as a graph network over the "
            "molecule itself — but needs more data and far more hyperparameter care than the alternatives.",
            "Support vector machine": "strong in high dimensions and can model non-linearity through "
            "kernels, at the cost of tuning and poor scaling to very large datasets.",
        }
        notes.append(f"**Model:** {model_choice} — {model_notes[model_choice]}")
        if model_choice in ("Logistic regression", "Support vector machine") and (
            "Scale/normalize numerical features" not in prepro_steps
        ):
            notes.append(
                "⚠️ **Inconsistency:** you chose a scale-sensitive model in Stage 3 but did not select "
                "feature scaling in Stage 2. Your Stage 2 and Stage 3 decisions have to agree."
            )

        # Stage 4
        if "External test set (different source)" in validation:
            notes.append(
                "**Validation:** an external test set is the strongest evidence you can offer — it is the "
                "only design that measures transfer to unseen chemical space and a different lab."
            )
        elif "Leave-cluster-out validation (e.g., by scaffold or class)" in validation:
            notes.append(
                "**Validation:** scaffold-split validation is the right instinct. It tests extrapolation to "
                "novel molecule types rather than recognition of close analogs."
            )
        elif "Cross-validation" in validation:
            notes.append(
                "**Validation:** cross-validation is a robust internal check, but molecular data is "
                "clustered — random folds will still overestimate performance on genuinely new scaffolds."
            )
        elif "Simple train/test split" in validation:
            notes.append(
                "⚠️ **Validation:** a single random split is the weakest option here. With clustered "
                "molecular data it mostly measures how well the model recognizes analogs it has already seen."
            )
        else:
            notes.append(
                "⚠️ **Validation:** no validation strategy was selected. Without one you have a model but "
                "no claim you can defend."
            )

        st.info("### Review of your lifecycle plan\n\n" + "\n\n".join(notes))
        st.caption(
            "Carry this plan forward — subsection 1.2 turns exactly these four decisions into a study "
            "design you have to defend."
        )

st.markdown(
    """
---
**Key takeaways**

- AI is a tool built by people to solve a stated problem. Every capability in the timeline arrived
  because someone framed a problem narrowly enough to make progress on it.
- A confident claim about AI is not evidence. Check what the system can actually do before you repeat it.
- **Every AI project makes the same four decisions** — data, preprocessing, model, validation — and each
  one constrains the next. Choosing a model before you know your data is choosing in the wrong order.
- Where the data comes from sets the ceiling on what the finished model can claim.
- **A validation strategy is the claim you are making.** With clustered data, a random split mostly
  measures how well the model recognises what it has already seen.

**Resources**
- [ChEMBL](https://www.ebi.ac.uk/chembl/) · [PubChem](https://pubchem.ncbi.nlm.nih.gov/)
- [RDKit](https://www.rdkit.org/) · [DeepChem](https://deepchem.io/) · [scikit-learn](https://scikit-learn.org/stable/)
"""
)
