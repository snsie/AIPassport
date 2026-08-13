import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from faker import Faker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import aipassport_config as cfg

# ── Track-specific framing (this file is the basic science track) ───────────
DATASET_NAME = "ImmPort (Immunology Database and Analysis Portal)"
DATASET_LINK = "https://www.immport.org/shared/home"
SUBJECT_TERM = "Donor"
SAMPLE_TERM = "Biological Specimen"
UNDERSERVED_EXAMPLE = "Donors of non-European ancestry"

st.markdown(
    f"""
Before you clean a dataset, you have to decide whether it is worth using at all. You are acting as the
**ethical compliance officer** reviewing a proposed study protocol using data from
[{DATASET_NAME}]({DATASET_LINK}), which uses human subject data from immunology, vaccine response, and
other biomedical studies.

Let's look at five different audits, each answering a different question about trust:

| Audit | The question it answers |
| --- | --- |
| **1. Consent** | Did the {SUBJECT_TERM.lower()} actually understand what they agreed to? |
| **2. Representation** | Who is missing from this data, and what would it take to reach them? |
| **3. Security** | If this data leaked tomorrow, what would it cost? |
| **4. Label quality** | Do the humans who annotated it agree with each other? |
| **5. Vocabulary** | Can anyone outside this lab read it? |
"""
)

AUDITS = [
    "1. Consent (Autonomy)",
    "2. Representation (Justice)",
    "3. Security (Privacy)",
    "4. Label Quality",
    "5. Common Vocabulary",
]
# A keyed segmented_control rather than st.tabs: tab selection lives in the browser and is
# lost whenever a widget inside a tab triggers a rerun, which is what sent learners back to
# the first activity mid-edit. This selection is in session_state, so it survives.
audit = st.segmented_control(
    "Audit",
    AUDITS,
    default=AUDITS[0],
    key="m3_audit",
    required=True,
)
# ═══════════════════════════════════════════════════════════════════════════
# Audit 1 — Consent
# ═══════════════════════════════════════════════════════════════════════════
if audit == AUDITS[0]:
    st.header("Audit 1: The 'Fine Print' Audit")
    st.markdown(
        f"""
    Informed consent is not just a signature on a document — it is a dynamic, ongoing dialogue. The current
    protocol uses standard legal text for {SUBJECT_TERM.lower()} consent. Use the tool below to revise the
    language and watch what happens to comprehension.
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Current protocol (legal standard)")
        st.warning(
            f"The undersigned {SUBJECT_TERM} hereby grants permissions for the indefinite utilization of "
            f"{SAMPLE_TERM}s and associated metadata, waiving all rights to pecuniary gain..."
        )
        st.markdown(
            "**Audit finding:** low comprehension. Donors may feel alienated or coerced. A signature "
            "obtained this way is legally valid and ethically thin — and it will not survive the donor "
            "learning what 'indefinite utilization' covered."
        )

    with col2:
        st.markdown("#### Revision tool")
        literacy = st.select_slider(
            "Language complexity level:",
            options=["Technical jargon", "Standard", "Simplified"],
            value="Technical jargon",
            key="m3_consent_literacy",
        )

        # Every level rewrites the text, including Standard. Showing revised wording only at the top of
        # the slider made the middle setting look broken — nothing on screen changed when it moved.
        if literacy == "Technical jargon":
            st.info("Status: no changes made — see the audit finding on the left.")
        elif literacy == "Standard":
            st.info(
                f"**Revised text:** 'You are being asked to permit the storage and research use of your "
                f"{SAMPLE_TERM.lower()} and related data. Participation is voluntary and you may withdraw "
                "your consent at a later date.'"
            )
            st.markdown(
                "**Audit result:** improved, but still transactional. The donor is informed; they are not "
                "yet a partner."
            )
        else:
            st.success(
                f"**Revised text:** 'We are asking for your permission to use your {SAMPLE_TERM.lower()} to "
                "help researchers understand how the immune system works. You can say no, and you can "
                "withdraw later. We want you to be a partner in this science.'"
            )
            st.markdown("**Audit result:** compliant. The donor can make an informed choice.")

# ═══════════════════════════════════════════════════════════════════════════
# Audit 2 — Representation
# ═══════════════════════════════════════════════════════════════════════════
if audit == AUDITS[1]:
    st.header("Audit 2: The 'Hidden Population' Audit")
    st.markdown(
        """
    The current recruitment plan is **passive** — an email blast, reaching whoever already answers emails
    from this institution. Use the REP-EQUITY toolkit below to name the missing group and select the active
    strategies that would close the gap.

    This matters more in basic science than it looks: a reference panel drawn from one ancestry produces
    variant-effect predictions that are simply wrong for everyone else.
    """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Step 1 — Define the underserved group**")
        st.text_input(
            f"Which group is missing from {DATASET_NAME}?",
            value=UNDERSERVED_EXAMPLE,
            key="m3_rep_group",
        )

        st.markdown("**Step 2 — Set the recruitment goal**")
        goal = st.slider(
            f"Target proportion of underserved {SUBJECT_TERM.lower()}s (%):",
            0,
            100,
            30,
            help="The share of the total sample needed from this group for adequate statistical power.",
            key="m3_rep_goal",
        )
        baseline = 5  # passive recruitment, measured

    with c2:
        st.markdown("**Step 3 — Select active strategies**")
        s1 = st.checkbox(
            "Community liaisons (+10%)",
            help="Trusted community members facilitate recruitment.",
            key="m3_rep_liaisons",
        )
        s2 = st.checkbox(
            "Translated materials (+5%)",
            help="Consent forms in the target group's own language.",
            key="m3_rep_translation",
        )
        s3 = st.checkbox(
            "Logistical support (+10%)",
            help="Transportation or mobile collection sites, removing access barriers.",
            key="m3_rep_logistics",
        )

        current = baseline + (10 if s1 else 0) + (5 if s2 else 0) + (10 if s3 else 0)

        rep_df = pd.DataFrame(
            {
                "Stage": ["Baseline (passive)", "With selected strategies", "Target goal"],
                "Percentage": [baseline, current, goal],
            }
        )
        fig = px.bar(
            rep_df,
            x="Stage",
            y="Percentage",
            color="Stage",
            color_discrete_map={
                "Baseline (passive)": cfg.MUTED,
                "With selected strategies": cfg.SUCCESS,
                "Target goal": cfg.OXFORD_BLUE,
            },
        )
        fig.update_layout(height=260, showlegend=False, margin=dict(l=40, r=20, t=25, b=40))
        st.plotly_chart(fig, width="stretch")

        with st.expander("View chart data as text (accessible alternative)"):
            st.dataframe(rep_df, width="stretch", hide_index=True)

    if current >= goal:
        st.success(
            f"**Audit result: PASS.** The selected strategies reach a representative sample "
            f"({current}% vs. a {goal}% target)."
        )
    else:
        st.warning(
            f"**Audit result: FAIL.** {current}% against a {goal}% target — additional strategies are "
            "needed. Note that every strategy here costs money and time; representation is a budget line, "
            "not a good intention."
        )

# ═══════════════════════════════════════════════════════════════════════════
# Audit 3 — Security
# ═══════════════════════════════════════════════════════════════════════════
if audit == AUDITS[2]:
    st.header("Audit 3: The 'Data Fortress' Audit")
    st.markdown(
        f"""
    Protecting {SUBJECT_TERM.lower()} privacy is a moral obligation before it is a legal one — and genomic
    data cannot be truly de-identified, because the sequence *is* the identifier. The protocol currently
    lacks depth. Select the security layers needed to protect the {SAMPLE_TERM.lower()} data; you must reach
    a **secure** rating to pass.
    """
    )

    c1, c2, c3, c4 = st.columns(4)
    l1 = c1.checkbox(
        "End-to-end encryption",
        help="Data is encoded so only authorized parties can read it.",
        key="m3_sec_encryption",
    )
    l2 = c2.checkbox(
        "Role-based access control",
        help="System access is restricted by the user's role.",
        key="m3_sec_rbac",
    )
    l3 = c3.checkbox(
        "De-identification",
        help="Personal identifiers are removed from the dataset.",
        key="m3_sec_deid",
    )
    l4 = c4.checkbox(
        "Regular audits",
        help="Routine checks to find and patch vulnerabilities.",
        key="m3_sec_audits",
    )

    score = sum([l1, l2, l3, l4])

    st.markdown("---")
    if score == 4:
        st.success("**Status: SECURE.** Multi-layered protocols are active. Compliance verified.")
    elif score >= 2:
        st.warning(
            "**Status: VULNERABLE.** Some protections are in place, but gaps remain. High risk of breach."
        )
    else:
        st.error("**Status: CRITICAL RISK.** Data is effectively unprotected. Protocol rejected.")

    st.caption(
        "De-identification alone is the most common failure here: it protects against casual browsing, not "
        "against re-identification from a rare genotype or a small donor pool."
    )

# ═══════════════════════════════════════════════════════════════════════════
# Audit 4 — Label quality
# ═══════════════════════════════════════════════════════════════════════════
if audit == AUDITS[3]:
    st.header("Audit 4: Do the Annotators Agree?")
    st.markdown(
        """
    **The case:** researchers annotate genomic variants as **pathogenic (1)** or **benign (0)**. A model
    trained on inconsistent annotations learns the inconsistency. Three things to establish:

    1. **Quantify agreement** — the intraclass correlation coefficient (ICC).
    2. **Find the ambiguous cases** — the variants worth a second opinion.
    3. **Decide how many annotators you need** — because expert time is the scarcest resource you have.
    """
    )

    c1, c2, c3 = st.columns(3)
    num_samples = c1.slider("Number of variants", 50, 1000, 200, key="m3_icc_samples")
    raters = c2.slider("Number of researchers", 2, 10, 5, key="m3_icc_raters")
    disagreement_rate = c3.slider(
        "Disagreement rate",
        0.0,
        0.5,
        0.2,
        help="Probability that any one researcher departs from the underlying truth.",
        key="m3_icc_disagreement",
    )

    @st.cache_data
    def simulate_researcher_annotations(n, n_raters, rate):
        """Simulated annotations. Ground truth is hidden from the annotators, as in reality."""
        rng = np.random.default_rng(42)
        ground_truth = rng.integers(0, 2, n)

        data = {"Variant_ID": np.arange(1, n + 1)}
        for i in range(1, n_raters + 1):
            ratings = ground_truth.copy()
            noise_mask = rng.random(n) < rate
            ratings[noise_mask] = 1 - ratings[noise_mask]
            data[f"Researcher_{i}"] = ratings

        return pd.DataFrame(data), ground_truth

    annotation_data, ground_truth = simulate_researcher_annotations(num_samples, raters, disagreement_rate)
    rating_cols = [c for c in annotation_data.columns if c.startswith("Researcher_")]
    numeric_data = annotation_data[rating_cols]

    st.markdown("**Simulated annotations** (0 = benign, 1 = pathogenic; one row per variant):")
    st.dataframe(annotation_data.head(), width="stretch")

    def icc(data):
        """Consistency ICC: between-item variance as a share of total variance."""
        mean_ratings = data.mean(axis=1)
        total_var = mean_ratings.var()
        within_item_var = data.var(axis=1).mean()
        return (total_var - within_item_var) / total_var if total_var > 0 else 0

    icc_value = icc(numeric_data)
    rater_word = "researcher"
    item_word = "variant"

    col_icc, col_dist = st.columns(2)
    with col_icc:
        st.subheader("Inter-rater reliability")
        st.metric(
            "ICC score",
            f"{icc_value:.4f}",
            help="Near 1.0 means the researchers agree. Lower values mean the label itself is noisy.",
        )
        if icc_value < 0:
            st.error(
                f"**A negative ICC ({icc_value:.3f}) is not a bug.** It means the {rater_word}s disagree with "
                f"each other *more* than the {item_word}s differ from one another — so the label carries "
                "essentially no signal about which "
                f"{item_word} is which. Training on this would teach a model to reproduce the disagreement."
            )
        elif icc_value < 0.5:
            st.error(
                "Poor agreement. No model can be more reliable than the labels it was trained on — this is "
                "a data problem, not a modelling problem."
            )
        elif icc_value < 0.75:
            st.warning("Moderate agreement. Expect a performance ceiling well below what the metrics promise.")
        else:
            st.success("Good agreement. The label is stable enough to train against.")

    with col_dist:
        st.subheader("Annotation distribution")
        dist_df = pd.DataFrame(
            {
                "Annotation": np.where(
                    numeric_data.values.flatten() == 1, "Pathogenic (1)", "Benign (0)"
                )
            }
        )
        fig2 = px.histogram(dist_df, x="Annotation", color="Annotation", title="Annotation counts")
        fig2.update_layout(height=320, showlegend=False, margin=dict(l=30, r=20, t=50, b=40))
        st.plotly_chart(fig2, width="stretch")

        with st.expander("View chart data as text (accessible alternative)"):
            st.dataframe(
                dist_df["Annotation"].value_counts().rename("Count").to_frame(),
                width="stretch",
            )

    st.subheader("High-disagreement variants")
    st.markdown(
        "The variants with the highest variance across annotators — split decisions. These are the "
        "candidates for expert re-review, and the cheapest available improvement to your dataset."
    )

    disagreement_view = annotation_data.assign(Disagreement_Score=numeric_data.var(axis=1))
    st.dataframe(
        disagreement_view.sort_values("Disagreement_Score", ascending=False).head(10),
        width="stretch",
    )

    st.subheader("How many annotators do you actually need?")
    st.markdown(
        "Read the curve left to right. At each point, that many researchers have annotated every sample "
        "and their majority vote becomes the label. A model is trained on those labels, and its accuracy "
        "is then checked against the true answer — which the simulation knows but no researcher saw. "
        "Where the curve stops climbing is where an extra annotator stops buying you anything."
    )

    @st.cache_data
    def accuracy_by_researcher_count(n, n_raters, rate):
        annotations, truth = simulate_researcher_annotations(n, n_raters, rate)
        cols = [c for c in annotations.columns if c.startswith("Researcher_")]
        accuracies = []
        for i, num in enumerate(range(1, len(cols) + 1)):
            y_consensus = (annotations[cols[:num]].mean(axis=1) > 0.5).astype(int)
            # Stand-in for sequence features: values that track the consensus label, plus noise
            rng = np.random.default_rng(i)
            X_features = rng.normal(loc=y_consensus.values[:, None], scale=1.5, size=(n, 10))
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, truth, test_size=0.3, random_state=42
            )
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            accuracies.append(accuracy_score(y_test, clf.predict(X_test)))
        return pd.DataFrame(
            {"Count of researchers": list(range(1, len(cols) + 1)), "Model accuracy": accuracies}
        )

    with st.spinner("Training one model per annotator count..."):
        results_df = accuracy_by_researcher_count(num_samples, raters, disagreement_rate)

    fig3 = px.line(
        results_df,
        x="Count of researchers",
        y="Model accuracy",
        markers=True,
        title="Model accuracy vs. number of annotators",
    )
    fig3.update_layout(height=380, margin=dict(l=40, r=20, t=55, b=45))
    st.plotly_chart(fig3, width="stretch")

    with st.expander("View chart data as text (accessible alternative)"):
        st.dataframe(results_df.round(3), width="stretch", hide_index=True)

    st.info(
        """
    **Interpretation**
    * **Diminishing returns.** The curve flattens. Its elbow is your answer to "how many annotators can we
      afford to stop at".
    * **Noise reduction.** More annotators means a more stable consensus, which means better training
      data — up to the point where the remaining error is in the biology, not the annotators.
    """
    )

# ═══════════════════════════════════════════════════════════════════════════
# Audit 5 — Common vocabulary
# ═══════════════════════════════════════════════════════════════════════════
if audit == AUDITS[4]:
    st.header("Audit 5: Can Anyone Else Read This Data?")
    st.markdown(
        """
    Your data can be consented, representative, secure, and reliably annotated — and still be useless to a
    collaborator, because it speaks only your lab's dialect. The **OMOP Common Data Model** is the fix:
    source values are replaced with standard **concept IDs** that mean the same thing everywhere.

    Below, synthetic donor-registry records from a biorepository are transformed into OMOP `person` and
    `condition_occurrence` tables. Watch what is removed and what is standardized — this is what has to
    happen before your specimen metadata can be pooled with another site's.
    """
    )

    c1, c2 = st.columns(2)
    num_patients = c1.slider("Number of donors", 5, 100, 10, key="m3_omop_n")
    seed_value = c2.number_input(
        "Random seed", value=123, help="Fixed for reproducibility.", key="m3_omop_seed"
    )

    @st.cache_data
    def generate_donor_source_records(n, seed):
        fake = Faker()
        Faker.seed(seed)
        random.seed(seed)

        races = ["White", "Black", "Asian", "Other"]
        ethnicities = ["Not Hispanic or Latino", "Hispanic or Latino"]
        rows = []
        for _ in range(n):
            gender = random.choice(["Male", "Female"])
            rows.append(
                {
                    "person_source_value": fake.unique.uuid4(),
                    "full_name": fake.name_male() if gender == "Male" else fake.name_female(),
                    "gender": gender,
                    "birthdate": fake.date_of_birth(minimum_age=18, maximum_age=90),
                    "address": fake.address(),
                    "race": random.choice(races),
                    "ethnicity": random.choice(ethnicities),
                }
            )
        df = pd.DataFrame(rows)
        df["birthdate"] = pd.to_datetime(df["birthdate"])
        return df

    source_data = generate_donor_source_records(num_patients, seed_value)

    st.subheader("Step 1 — Raw source data")
    st.markdown("Messy, non-standardized donor records exported from the biorepository's local database.")
    st.dataframe(source_data, width="stretch")

    GENDER_CONCEPTS = {"Male": 8507, "Female": 8532}
    RACE_CONCEPTS = {"White": 8527, "Black": 8516, "Asian": 8515, "Other": 8529}
    ETHNICITY_CONCEPTS = {"Not Hispanic or Latino": 38070399, "Hispanic or Latino": 38003563}
    ICD_TO_OMOP = {"E11.9": 201826, "I10": 320128, "J45.909": 317009, "F32.9": 440383}

    omop_person = pd.DataFrame(
        {
            "person_id": range(1, len(source_data) + 1),
            "gender_concept_id": source_data["gender"].map(GENDER_CONCEPTS).fillna(0).astype(int),
            "year_of_birth": source_data["birthdate"].dt.year,
            "month_of_birth": source_data["birthdate"].dt.month,
            "day_of_birth": source_data["birthdate"].dt.day,
            "race_concept_id": source_data["race"].map(RACE_CONCEPTS).fillna(0).astype(int),
            "ethnicity_concept_id": source_data["ethnicity"]
            .map(ETHNICITY_CONCEPTS)
            .fillna(0)
            .astype(int),
            "person_source_value": source_data["person_source_value"],
            "gender_source_value": source_data["gender"],
            "race_source_value": source_data["race"],
            "ethnicity_source_value": source_data["ethnicity"],
        }
    )

    st.subheader("Step 2 — Standardized OMOP `person` table")
    st.markdown(
        "Names and addresses are gone. Demographics are now **concept IDs** — `8507` means male in every "
        "OMOP database on earth, whether the source system wrote 'Male', 'M', or '1'."
    )
    st.dataframe(omop_person, width="stretch")

    with st.expander("The concept map behind Step 2"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Source value": list(GENDER_CONCEPTS) + list(RACE_CONCEPTS) + list(ETHNICITY_CONCEPTS),
                    "OMOP concept ID": list(GENDER_CONCEPTS.values())
                    + list(RACE_CONCEPTS.values())
                    + list(ETHNICITY_CONCEPTS.values()),
                    "Domain": ["Gender"] * len(GENDER_CONCEPTS)
                    + ["Race"] * len(RACE_CONCEPTS)
                    + ["Ethnicity"] * len(ETHNICITY_CONCEPTS),
                }
            ),
            width="stretch",
        )

    @st.cache_data
    def generate_donor_conditions(person_ids, seed):
        fake = Faker()
        Faker.seed(seed)
        random.seed(seed + 1)
        icd_codes = list(ICD_TO_OMOP)
        conditions = []
        for person_id in person_ids:
            for _ in range(random.randint(1, 3)):
                icd = random.choice(icd_codes)
                conditions.append(
                    {
                        "condition_occurrence_id": len(conditions) + 1,
                        "person_id": person_id,
                        "condition_concept_id": ICD_TO_OMOP[icd],
                        "condition_start_date": fake.date_between(start_date="-5y", end_date="-6m"),
                        "condition_type_concept_id": 32020,  # source problem list
                        "condition_source_value": icd,
                    }
                )
        return pd.DataFrame(conditions)

    st.subheader("Step 3 — `condition_occurrence` from ICD-10 codes")
    st.markdown(
        "The donor's recorded clinical history arrives as ICD-10 source codes and is mapped to OMOP "
        "standard concepts. A query written against `condition_concept_id = 201826` finds type-2 diabetes "
        "at a site that codes in SNOMED too — which is how you select comparable donors across cohorts."
    )
    st.dataframe(
        generate_donor_conditions(tuple(omop_person["person_id"]), int(seed_value)),
        width="stretch",
    )

st.markdown(
    """
---
**Key takeaways**

- Consent is comprehension, not a signature. The revision is cheap; the trust is not.
- Representation is a budget line. Passive recruitment reproduces whoever already answers your emails.
- De-identification is one layer of four, and the weakest one on its own — especially for sequence data.
- **A model cannot be more reliable than its labels.** Measure agreement before you measure accuracy.
- A standard vocabulary is what makes your data usable by anyone who did not build it.

**Resources:** [OHDSI OMOP CDM documentation](https://www.ohdsi.org/data-standardization/the-common-data-model/) ·
[REP-EQUITY Toolkit](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10719102/) ·
[{dataset}]({link})
""".format(dataset=DATASET_NAME, link=DATASET_LINK)
)
