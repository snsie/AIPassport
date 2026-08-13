import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
import aipassport_config as cfg

st.markdown(
    """
Subsection 2.1 reasoned about harm. This one **measures** it.

* **Activity 1 — Drift.** Watch a sepsis model decay under covariate shift, and test whether retraining fixes it.
* **Activity 2 — Calibration vs. discrimination.** Two different things a "good" model can be, and why AUC alone hides the gap.
* **Activity 3 — Subgroup performance.** Split a real ICU cohort by sex and find out whether one model serves both.
* **Activity 4 — Model card.** Document intended use and limits, so the next person inherits your findings instead of repeating them.
"""
)

# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data generators (Activities 1–2)
# ═══════════════════════════════════════════════════════════════════════════


@st.cache_data
def generate_sepsis_data(n=1000, drift_severity=0.0):
    np.random.seed(42)
    age = np.random.normal(65, 12, n).astype(int)
    # Lactate: normal < 2, sepsis > 2
    lactate = np.random.gamma(2, 1.5, n)
    wbc = np.random.normal(10, 3, n)

    # COVARIATE SHIFT: shift the input distributions (a lab-method change, or a sicker population)
    lactate = lactate + (drift_severity * 2.0)
    wbc = wbc + (drift_severity * 1.5)

    logits = -5 + (0.05 * age) + (0.8 * lactate) + (0.1 * wbc)
    probs = 1 / (1 + np.exp(-logits))
    sepsis = np.random.binomial(1, probs)

    return pd.DataFrame({"Age": age, "Lactate": lactate, "WBC": wbc, "Sepsis": sepsis})


@st.cache_data
def generate_vendor_data(n=2000, difference_factor=0.0):
    np.random.seed(101)
    # The local population shifts towards higher comorbidity and lower income
    comorb = np.random.normal(5 + difference_factor * 3, 2, n)
    income = np.random.normal(60 + difference_factor * -25, 15, n)
    comorb = np.clip(comorb, 0, 15)

    logits = -4 + (0.4 * comorb) - (0.02 * income)
    probs = 1 / (1 + np.exp(-logits))
    readmit = np.random.binomial(1, probs)

    return pd.DataFrame(
        {"Comorbidity_Index": comorb, "Income": income, "Readmission": readmit}
    )


# ═══════════════════════════════════════════════════════════════════════════
# Real eICU cohort (Activity 3)
# ═══════════════════════════════════════════════════════════════════════════

EICU_VARS = [
    "age", "height", "weight_admission",
    "lab_bun", "lab_hct", "lab_hgb", "lab_mch", "lab_mchc", "lab_mcv", "lab_rbc",
    "lab_rdw", "lab_albumin", "lab_bicarbonate", "lab_calcium", "lab_chloride",
    "lab_creatinine", "lab_glucose", "lab_platelets", "lab_potassium", "lab_sodium",
    "lab_wbc",
]


@st.cache_data
def load_eicu_cohort():
    """Bundled eICU demo extract. No network access: the CSV ships with the repository."""
    df = pd.read_csv("assets/datasets/csv/eicu_demo.csv")
    if "weight" in df.columns and "weight_admission" not in df.columns:
        df = df.rename(columns={"weight": "weight_admission"})
    return df


@st.cache_data
def get_processed_data(df_raw):
    """One-hot encode the categoricals and mean-impute the numeric columns."""
    x = df_raw.drop(
        columns=[
            "patient_id", "hospital_id", "admission_id", "admission_year",
            "weight_discharge", "discharge_location",
        ]
    )

    x[EICU_VARS + ["hospital_teaching"]] = x[EICU_VARS + ["hospital_teaching"]].apply(
        pd.to_numeric, errors="coerce", axis=1
    )
    x[EICU_VARS + ["hospital_teaching"]] = x[EICU_VARS + ["hospital_teaching"]].fillna(
        x[EICU_VARS + ["hospital_teaching"]].mean()
    )

    x = pd.get_dummies(
        x,
        columns=["ethnicity", "admission_source", "hospital_region", "hospital_beds"],
        dtype="int",
    )
    x["in_hospital_mortality"] = x["in_hospital_mortality"].fillna(0)
    return x


# ═══════════════════════════════════════════════════════════════════════════
# Main interface
# ═══════════════════════════════════════════════════════════════════════════

ACTIVITIES = [
    "1. Drift (Covariate Shift)",
    "2. Calibration vs. Discrimination",
    "3. Subgroup Performance",
    "4. Model Card Builder",
]
# A keyed segmented_control rather than st.tabs: tab selection lives in the browser and is
# lost whenever a widget inside a tab triggers a rerun, which is what sent learners back to
# the first activity mid-edit. This selection is in session_state, so it survives.
activity = st.segmented_control(
    "Activity",
    ACTIVITIES,
    default=ACTIVITIES[0],
    key="m2_activity",
    required=True,
)
# ── Activity 1: drift ──────────────────────────────────────────────────────
if activity == ACTIVITIES[0]:
    st.header("Activity 1: Recognizing Data Drift")
    st.markdown(
        "Simulate how model performance degrades over time under covariate shift, and test whether "
        "retraining recovers it."
    )

    col_sim, col_viz = st.columns([1, 2])

    with col_sim:
        st.subheader("Simulation controls")
        months = st.slider("Time since deployment (months)", 0, 12, 0, key="m2_drift_months")
        drift_sev = months / 10.0

        # 1. Baseline model, fitted at month 0
        df_train = generate_sepsis_data(n=1000, drift_severity=0.0)
        model_orig = LogisticRegression()
        model_orig.fit(df_train[["Age", "Lactate", "WBC"]], df_train["Sepsis"])

        # 2. Today's patients, drifted
        df_current = generate_sepsis_data(n=500, drift_severity=drift_sev)

        # 3. Mitigation strategy
        strategy = st.radio(
            "Mitigation strategy:",
            ["Do nothing", "Retrain model (refit on current data)"],
            key="m2_drift_strategy",
        )

        if strategy == "Do nothing":
            model_used = model_orig
        else:
            model_retrained = LogisticRegression()
            model_retrained.fit(df_current[["Age", "Lactate", "WBC"]], df_current["Sepsis"])
            model_used = model_retrained
            st.success("Model has learned the new normal (retrained).")

        preds = model_used.predict(df_current[["Age", "Lactate", "WBC"]])
        tn, fp, fn, tp = confusion_matrix(df_current["Sepsis"], preds).ravel()
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        st.divider()
        st.metric(
            "False positive rate",
            f"{fp_rate:.1%}",
            delta=f"{fp_rate - 0.05:.1%}",
            delta_color="inverse",
        )
        st.caption(
            "A high false positive rate leads to alert fatigue and unnecessary antibiotics — the alarm "
            "stops meaning anything."
        )

    with col_viz:
        st.subheader("Visualizing covariate shift")

        dist_df = pd.concat(
            [
                pd.DataFrame({"Lactate": df_train["Lactate"], "Dataset": "Original training data"}),
                pd.DataFrame({"Lactate": df_current["Lactate"], "Dataset": "Current patient data"}),
            ],
            ignore_index=True,
        )
        fig = px.histogram(
            dist_df,
            x="Lactate",
            color="Dataset",
            nbins=35,
            histnorm="probability density",
            opacity=0.55,
            barmode="overlay",
            title=f"Lactate distribution shift (month {months})",
        )
        fig.update_layout(height=420, margin=dict(l=40, r=20, t=55, b=45))
        st.plotly_chart(fig, width="stretch")

        with st.expander("View chart data as text (accessible alternative)"):
            st.markdown("**Lactate distribution in each dataset**")
            st.dataframe(
                dist_df.groupby("Dataset")["Lactate"].describe(), width="stretch"
            )

        st.info(
            "This is **covariate shift**: the input distribution has moved while the model's learned "
            "decision boundary has not. Nothing is broken in the code — the world changed."
        )

# ── Activity 2: calibration vs discrimination ──────────────────────────────
if activity == ACTIVITIES[1]:
    st.header("Activity 2: Calibration vs. Discrimination")
    st.markdown(
        "A vendor sells you a readmission model with an excellent AUC. Analyze its performance on **your** "
        "population by separating discrimination (does it rank patients correctly?) from calibration "
        "(does 0.7 actually mean 70%?)."
    )

    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Vendor model check")
        diff_factor = st.slider("Population mismatch", 0.0, 1.0, 0.6, key="m2_cal_mismatch")

        df_vendor = generate_vendor_data(n=2000, difference_factor=0.0)
        df_local = generate_vendor_data(n=1000, difference_factor=diff_factor)

        model_vendor = LogisticRegression()
        model_vendor.fit(df_vendor[["Comorbidity_Index", "Income"]], df_vendor["Readmission"])

        local_probs = model_vendor.predict_proba(df_local[["Comorbidity_Index", "Income"]])[:, 1]

        auc = roc_auc_score(df_local["Readmission"], local_probs)
        st.metric("Discrimination (AUC)", f"{auc:.3f}")
        if auc > 0.85:
            st.success("High discrimination — good at ranking patients by risk.")
        else:
            st.warning("Low discrimination.")

        st.caption(
            "Now turn the mismatch slider up and watch what happens to the curve on the right while this "
            "number barely moves."
        )

    with c2:
        st.subheader("Calibration curve (reliability diagram)")

        prob_true, prob_pred = calibration_curve(df_local["Readmission"], local_probs, n_bins=10)

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color=cfg.MUTED, dash="dash"),
                name="Perfectly calibrated",
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode="lines+markers",
                line=dict(color=cfg.CHART_PRIMARY),
                name="Vendor model",
            )
        )
        fig2.update_layout(
            title="Calibration curve",
            xaxis_title="Predicted probability",
            yaxis_title="Observed risk",
            height=430,
            margin=dict(l=40, r=20, t=55, b=45),
        )
        st.plotly_chart(fig2, width="stretch")

        with st.expander("View chart data as text (accessible alternative)"):
            st.markdown(
                "Each row is one bin of predictions. A perfectly calibrated model would have the two "
                "columns equal on every row."
            )
            st.dataframe(
                pd.DataFrame(
                    {"Predicted probability": prob_pred, "Observed risk": prob_true}
                ).round(3),
                width="stretch",
                hide_index=True,
            )

        st.info(
            """
        **How to read this:**
        * **On the diagonal** — perfect calibration.
        * **Below the diagonal** — the model **overestimates** risk (predicted 80%, observed 40%).
        * **Above the diagonal** — the model **underestimates** risk.

        A model can rank every patient correctly and still be badly wrong about the numbers. If a care
        pathway triggers at "risk > 20%", it is the calibration that decides who gets treated.
        """
        )

# ── Activity 3: subgroup performance ───────────────────────────────────────
if activity == ACTIVITIES[2]:
    st.header("Activity 3: Subgroup Performance")
    st.markdown(
        """
    2.1 argued that a model can help most people while harming some. Here you check, on a real cohort.

    The data is the bundled **eICU demo extract** — ICU admissions with demographics and admission labs,
    and in-hospital mortality as the outcome. Clinical data often contains meaningful differences between
    males and females; pooling everyone into one analysis averages those differences away.
    """
    )

    with st.spinner("Loading cohort..."):
        df_raw = load_eicu_cohort()
        df_processed = get_processed_data(df_raw)
        if "gender" not in df_processed.columns:
            df_processed["gender"] = df_raw["gender"]

    st.caption(
        f"{df_processed.shape[0]:,} admissions · {df_processed.shape[1]:,} columns after one-hot encoding "
        f"and mean imputation."
    )

    st.subheader("3.1 Do the sexes look different before any model is fitted?")

    selected_variables = st.multiselect(
        "Variables to compare:",
        EICU_VARS,
        default=["lab_bun", "lab_hct", "lab_hgb"],
        help="Mean value in each sex, split by whether the patient survived.",
        key="m2_sub_vars",
    )

    if selected_variables:
        rows = []
        for sex in ("Female", "Male"):
            df_sex = df_processed[df_processed["gender"] == sex]
            for outcome_value, outcome_label in ((0, "Survived"), (1, "In-hospital mortality")):
                subset = df_sex[df_sex["in_hospital_mortality"] == outcome_value]
                for var in selected_variables:
                    rows.append(
                        {
                            "Variable": var,
                            "Group": f"{sex} – {outcome_label}",
                            "Mean value": subset[var].mean(),
                        }
                    )
        means_df = pd.DataFrame(rows)

        fig3 = px.bar(
            means_df,
            x="Mean value",
            y="Variable",
            color="Group",
            orientation="h",
            barmode="group",
            title="Mean values by sex and mortality status",
        )
        fig3.update_layout(
            height=max(420, 90 * len(selected_variables)), margin=dict(l=40, r=20, t=55, b=45)
        )
        st.plotly_chart(fig3, width="stretch")

        with st.expander("View chart data as text (accessible alternative)"):
            st.dataframe(
                means_df.pivot(index="Variable", columns="Group", values="Mean value"),
                width="stretch",
            )
    else:
        st.info("Select at least one variable to compare.")

    st.divider()
    st.subheader("Question 1")
    st.markdown("Why is it important to analyze clinical data separately for males and females before modeling?")

    q1_options = {
        "A": "To reduce the number of observations",
        "B": "To make the dataset more complex",
        "C": "To identify sex-specific patterns that might be masked in pooled data",
        "D": "To apply the same model to both groups without changes",
    }
    q1_choice = st.radio(
        "Select answer:",
        list(q1_options.keys()),
        format_func=lambda x: f"{x}) {q1_options[x]}",
        key="m2_sub_q1",
    )

    if st.button("Submit Question 1", key="m2_sub_q1_btn"):
        if q1_choice == "C":
            st.success(
                "Correct. Clinical data often contains meaningful differences between males and females. "
                "Analyzing the whole dataset as a single group averages those differences out or hides "
                "them entirely."
            )
        else:
            st.error("Not quite — try again.")

    st.divider()
    st.subheader("3.2 Does one model serve both groups equally?")
    st.markdown(
        """
    Now fit **three** multivariate logistic regressions — female-only, male-only, and the whole cohort —
    and compare their AUROC on held-out patients. This is the measurement that 2.1's justice argument was
    reaching for.
    """
    )

    selected_predictors = st.multiselect(
        "Predictors:",
        EICU_VARS,
        default=EICU_VARS,
        help="Variables entered into the multivariate logistic regression.",
        key="m2_sub_predictors",
    )

    if st.button("Train and evaluate by sex", key="m2_sub_train_btn"):
        if not selected_predictors:
            st.error("Select at least one predictor.")
        else:
            with st.spinner("Fitting three models..."):
                df_female = df_processed[df_processed["gender"] == "Female"]
                df_male = df_processed[df_processed["gender"] == "Male"]

                formula = "in_hospital_mortality ~ " + " + ".join(selected_predictors)
                results = {}
                try:
                    for label, subset in (
                        ("Female", df_female),
                        ("Male", df_male),
                        ("All cohort", df_processed),
                    ):
                        train_dat, test_dat = train_test_split(
                            subset, test_size=0.2, random_state=2025
                        )
                        reg = smf.logit(formula, data=train_dat).fit(disp=0)
                        results[label] = roc_auc_score(
                            test_dat["in_hospital_mortality"],
                            reg.predict(test_dat[selected_predictors]),
                        )
                except Exception as e:
                    results = {}
                    st.error(f"Error training model: {e}")

            if results:
                res_df = pd.DataFrame(
                    {"Group": list(results.keys()), "AUROC": list(results.values())}
                )
                fig4 = px.bar(
                    res_df,
                    x="Group",
                    y="AUROC",
                    color="Group",
                    text=res_df["AUROC"].map(lambda value: f"{value:.3f}"),
                    title="Multivariate model performance by sex",
                    range_y=[0.5, 1.0],
                )
                fig4.update_layout(
                    height=430, showlegend=False, margin=dict(l=40, r=20, t=55, b=45)
                )
                st.plotly_chart(fig4, width="stretch")

                with st.expander("View chart data as text (accessible alternative)"):
                    st.dataframe(
                        res_df.round(3), width="stretch", hide_index=True
                    )

                gap = abs(results["Female"] - results["Male"])
                st.metric("AUROC gap between sexes", f"{gap:.3f}")
                st.caption(
                    "A gap here is the measurable form of the fairness concern from 2.1. It is also the "
                    "number that belongs in the model card's limitations section."
                )

                with st.expander("View results as a table"):
                    st.dataframe(res_df, width="stretch")

    st.divider()
    st.subheader("Question 2")
    st.markdown(
        "When comparing the outcomes of these sex-specific models, what are we most interested in identifying?"
    )

    q2_options = {
        "A": "The variables with the highest missing values",
        "B": "The computational time for each model",
        "C": "Any performance gaps or patterns that differ between sexes",
        "D": "Whether the models have the same coefficients",
    }
    q2_choice = st.radio(
        "Select answer:",
        list(q2_options.keys()),
        format_func=lambda x: f"{x}) {q2_options[x]}",
        key="m2_sub_q2",
    )

    if st.button("Submit Question 2", key="m2_sub_q2_btn"):
        if q2_choice == "C":
            st.success(
                "Correct. The main goal is to see whether the model performs differently across the two "
                "groups. Are predictions more accurate for one sex than the other? Those performance "
                "differences point to underlying biological, clinical, or systemic factors."
            )
        else:
            st.error("Not quite — try again.")

# ── Activity 4: model card ─────────────────────────────────────────────────
if activity == ACTIVITIES[3]:
    st.header("Activity 4: Model Card Builder")
    st.markdown(
        """
    Everything you measured in Activities 1–3 is worthless to the next person unless it is written down.
    A **model card** is the artifact that carries intended use, performance, and limits forward — including
    the drift you found, the calibration gap, and the subgroup AUROC difference.

    You will build a fuller version of this document in Module 7.
    """
    )

    col_input, col_card = st.columns(2)

    with col_input:
        st.subheader("Model details")
        mc_name = st.text_input("Model name", "Sepsis Prediction v1.0", key="m2_card_name")
        mc_dev = st.text_input("Developer", "Hospital AI Team", key="m2_card_dev")
        mc_users = st.text_area(
            "Intended users", "Emergency Department triage nurses", key="m2_card_users"
        )
        mc_limits = st.text_area(
            "Caveats / limitations",
            "Not validated for pediatric patients or those with pre-existing immunosuppression.",
            key="m2_card_limits",
        )
        mc_ethics = st.text_area(
            "Ethical considerations",
            "Training data was drawn primarily from Region A; potential bias against Region B demographics.",
            key="m2_card_ethics",
        )
        mc_monitoring = st.text_area(
            "Monitoring plan",
            "Track the false positive rate monthly against the deployment baseline; re-evaluate calibration "
            "and subgroup AUROC quarterly.",
            key="m2_card_monitoring",
        )

    with col_card:
        st.subheader("Preview")
        st.markdown(
            f"""
        <div style="background-color:{cfg.SURFACE_ALT}; padding:20px; border-radius:10px; border:1px solid {cfg.BORDER};">
            <h3>Model Card: {mc_name}</h3>
            <p><strong>Developer:</strong> {mc_dev}</p>
            <hr>
            <h4>1. Intended Use</h4>
            <p>{mc_users}</p>
            <h4>2. Performance Metrics</h4>
            <p><strong>Discrimination:</strong> AUC, reported overall and by sex</p>
            <p><strong>Reliability:</strong> calibration slope and reliability diagram</p>
            <p><strong>Operating point:</strong> false positive rate at the deployed threshold</p>
            <h4>3. Caveats &amp; Limitations</h4>
            <p style="color:{cfg.DANGER};">{mc_limits}</p>
            <h4>4. Ethical Considerations</h4>
            <p>{mc_ethics}</p>
            <h4>5. Monitoring</h4>
            <p>{mc_monitoring}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
---
**Key takeaways**

- **Drift** is not a bug. The model is unchanged; the population moved. Only monitoring detects it.
- **Discrimination and calibration are different claims.** A high AUC says nothing about whether the
  probabilities are trustworthy, and care pathways act on probabilities.
- **Aggregate performance hides subgroup performance.** Report it split, or you have not reported it.
- A model card turns all of the above from a finding into an inherited constraint.

**Resources:** [eICU Collaborative Research Database](https://eicu-crd.mit.edu/) ·
[scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
"""
)
