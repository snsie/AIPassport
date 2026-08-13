import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.markdown(
    """
You are a clinical informatics researcher at a large academic medical center. Your hospital has
higher-than-expected **30-day readmission rates for congestive heart failure (CHF) patients**, and you have
been asked to design an AI study to predict readmission risk and support discharge planning.

A design is only worth as much as the rigor behind it. This subsection moves in four steps:

1. **Write the design brief** — the gap, the question, and the data plan.
2. **Do the rigor work** — detect and handle outliers in a real table, and see what your choice does to the numbers.
3. **Commit to a validation strategy** — splitting, cross-validation, external validation, subgroup performance.
4. **Carry the decision to your team** — one professional message that a busy senior colleague will actually read.

**Resources.** The two databases below are the kind of data a study like this runs on, and the two tools
are what you would build it with — worth opening once now so the data plan in Part 1 is concrete rather
than hypothetical:
[MIMIC-IV](https://physionet.org/content/mimiciv/) and
[eICU](https://eicu-crd.mit.edu/) (de-identified ICU records, free with training and a data-use
agreement) · [Google Colab](https://colab.research.google.com/) and
[scikit-learn](https://scikit-learn.org/stable/) (where the analysis would actually run).
"""
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — The design brief
# ═══════════════════════════════════════════════════════════════════════════
st.header("1. The Design Brief")

st.subheader("1.1 The gap and the question")
st.text_area(
    "**Gap.** What are some potential limitations of predictive models? Consider which limitations would "
    "be most relevant as you design an AI study to predict readmission risk and support discharge "
    "planning for congestive heart failure.",
    key="m1_design_gap",
)
st.text_area(
    "**Question.** State one primary research question that closes that gap, using SMART criteria "
    "(specific, measurable, achievable, relevant, time-bound).",
    key="m1_design_question",
)

st.subheader("1.2 The data plan")
data_elements = st.multiselect(
    "**Elements.** Which MIMIC-IV data elements will your model use?",
    [
        "Demographics",
        "Diagnoses (ICD codes)",
        "Procedures",
        "Medications",
        "Laboratory values",
        "Clinical notes",
        "Vitals",
    ],
    key="m1_design_elements",
)
st.text_area(
    "**Cohort.** Give your inclusion and exclusion criteria in two or three lines.",
    key="m1_design_cohort",
)
st.text_area(
    "**Missingness and bias.** How will you handle missing values, and what bias in this data are you "
    "most worried about carrying into the model?",
    key="m1_design_missing",
)
st.text_area(
    "**Preprocessing.** Name the transformations you will apply — lab-value normalization, temporal "
    "aggregation over the stay, any clinically derived variables.",
    key="m1_design_prepro",
)

if data_elements:
    st.caption(f"Your model will see: {', '.join(data_elements)}.")
    if "Clinical notes" in data_elements:
        st.caption(
            "Note: including clinical notes commits you to a text-feature pipeline — and to a "
            "de-identification review before the data leaves the EHR."
        )

# There is no answer key for a design brief — a good gap and a good cohort depend on the study. What can
# be checked is whether each answer does the job the brief needs it to do, so the self-check below is
# written as criteria the learner applies to their own text rather than as a model answer to copy.
with st.expander("Check your own answers against these criteria"):
    st.markdown(
        """
    A design brief has no single correct answer, but a defensible one clears all six bars below. Reread
    what you wrote and mark each one honestly.

    | Input | Your answer clears the bar if… |
    | --- | --- |
    | **Gap** | It names a limitation you could show evidence for — a population the model was never validated on, a data type it ignores, a horizon it cannot see — not "more research is needed". |
    | **Question** | A reader can tell what you will measure, in whom, and by when. If it has no measurable outcome, it is a topic, not a question. |
    | **Elements** | Every element you ticked is plausibly recorded *before* discharge. Anything recorded after the decision point is leakage, not a feature. |
    | **Cohort** | Someone else could apply your criteria to the same database and pull the same patients. |
    | **Missingness and bias** | You named a handling rule *and* a named bias — and said which direction the bias would push the model, not just that it exists. |
    | **Preprocessing** | Each transformation is one you could compute from the elements you selected above. |

    The most common failure is a question the data plan cannot answer. If your question mentions
    something your selected elements never record, one of the two has to change.
    """
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — The rigor lab (outliers)
# ═══════════════════════════════════════════════════════════════════════════
st.header("2. Rigor Lab: Outliers in Your Cohort")

st.markdown(
    """
Your brief promised to handle missing and extreme values. This is where you actually do it.

Below is a **simulated CHF admission table** of 40 patients:

- `age` — age of patient (years)
- `length_of_stay` — duration of hospitalization (days)
- `bnp` — admission B-type Natriuretic Peptide (pg/mL)
- `sodium` — admission sodium (mmol/L)
- `readmit_30d` — 1 = readmitted within 30 days, 0 = no
"""
)


@st.cache_data
def load_chf_cohort():
    rng = np.random.default_rng(42)
    n_patients = 40
    df = pd.DataFrame(
        {
            "patient_id": np.arange(1, n_patients + 1),
            "age": np.append(rng.normal(68, 11, n_patients - 1), [105]),                 # one high outlier
            "length_of_stay": np.append(rng.exponential(4, n_patients - 2), [30, 0.2]),  # two outliers
            "bnp": np.append(rng.normal(900, 500, n_patients - 1), [9000]),              # one extreme
            "sodium": np.append(rng.normal(137, 5, n_patients - 1), [110]),              # one low
            "readmit_30d": rng.binomial(1, 0.36, n_patients),
        }
    )
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


df = load_chf_cohort()
VARIABLES = ["age", "length_of_stay", "bnp", "sodium"]

st.dataframe(df, width="stretch")


def iqr_bounds(column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr, q1, q3, iqr


st.subheader("2.1 See them")
st.markdown("A boxplot makes an extreme value obvious.")

sel_plot = st.selectbox("Variable for boxplot:", VARIABLES, key="m1_rigor_plot_var")
fig = px.box(df, x=sel_plot, points="all", hover_data=["patient_id"])
fig.update_layout(
    height=320,
    xaxis_title=sel_plot,
    yaxis_title="",
    margin=dict(l=40, r=20, t=25, b=35),
)
st.plotly_chart(fig, width="stretch")

with st.expander("View chart data as text (accessible alternative)"):
    st.markdown(f"**Five-number summary of `{sel_plot}`**")
    st.dataframe(df[sel_plot].describe().to_frame().T, width="stretch")
    st.markdown("**The five most extreme values, furthest from the median first**")
    st.dataframe(
        df.assign(distance_from_median=(df[sel_plot] - df[sel_plot].median()).abs())
        .sort_values("distance_from_median", ascending=False)
        .head(5)[["patient_id", sel_plot]],
        width="stretch",
        hide_index=True,
    )

st.text_area(
    "Which points look like outliers, and which patient IDs are they?", key="m1_rigor_visual_notes"
)

st.subheader("2.2 Measure them")
st.markdown(
    "The 1.5×IQR rule flags any value above Q3 + 1.5×IQR or below Q1 − 1.5×IQR. It is a convention, not "
    "a law — but it is a convention you can write down in a methods section."
)

sel_stat = st.selectbox("Variable for threshold calculation:", VARIABLES, key="m1_rigor_calc_var")
lower, upper, q1, q3, iqr = iqr_bounds(sel_stat)

bound_cols = st.columns(3)
bound_cols[0].metric("IQR", f"{iqr:.1f}", help=f"Q1 = {q1:.1f}, Q3 = {q3:.1f}")
bound_cols[1].metric("Lower bound", f"{lower:.1f}")
bound_cols[2].metric("Upper bound", f"{upper:.1f}")

outlier_mask = (df[sel_stat] < lower) | (df[sel_stat] > upper)
st.markdown("**Rows flagged by the IQR rule:**")
st.dataframe(df[outlier_mask][["patient_id", sel_stat]], width="stretch")
st.text_area(
    "Do the flagged values look like data-entry errors, or like real patients who are genuinely unusual? "
    "Your answer changes what you are allowed to do next.",
    key="m1_rigor_flagged_notes",
)

st.subheader("2.3 See what they cost you")
sel_compare = st.selectbox("Variable for comparison:", VARIABLES, key="m1_rigor_compare_var")
lwr2, upr2, *_ = iqr_bounds(sel_compare)
with_out = df[sel_compare]
wout_out = df[~((df[sel_compare] < lwr2) | (df[sel_compare] > upr2))][sel_compare]

comp_cols = st.columns(2)
with comp_cols[0]:
    st.markdown("**All data**")
    st.write(f"Mean: {with_out.mean():.2f}")
    st.write(f"Std: {with_out.std():.2f}")
    st.write(f"Median: {with_out.median():.2f}")
with comp_cols[1]:
    st.markdown("**Outliers excluded**")
    st.write(f"Mean: {wout_out.mean():.2f}")
    st.write(f"Std: {wout_out.std():.2f}")
    st.write(f"Median: {wout_out.median():.2f}")

st.text_area(
    "Which statistic moved most — mean, standard deviation, or median? Why does that matter when the "
    "number ends up in a paper?",
    key="m1_rigor_effect_notes",
)

st.subheader("2.4 Handle them")
st.markdown(
    """
Three defensible strategies, each with a different cost:

- **Remove** — honest about uncertainty, but throws away real patients and shrinks your cohort.
- **Winsorize** — keeps every row, at the price of a value that was never measured.
- **Impute with median** — keeps the row and the sample size, and erases the signal that made it unusual.
"""
)

sel_handle = st.selectbox("Variable for handling strategies:", VARIABLES, key="m1_rigor_handle_var")
approach = st.radio(
    "Strategy:",
    ["Remove (exclude outlier rows)", "Winsorize (cap at threshold)", "Impute with median"],
    key="m1_rigor_approach",
)
lwr, upr, *_ = iqr_bounds(sel_handle)
series = df[sel_handle]

if approach.startswith("Remove"):
    handled = series[(series >= lwr) & (series <= upr)]
elif approach.startswith("Winsor"):
    handled = series.clip(lwr, upr)
else:
    median = series[(series >= lwr) & (series <= upr)].median()
    handled = series.copy()
    handled[(handled < lwr) | (handled > upr)] = median

handle_cols = st.columns(3)
handle_cols[0].metric("Mean", f"{handled.mean():.2f}", f"{handled.mean() - series.mean():+.2f}")
handle_cols[1].metric("Std", f"{handled.std():.2f}", f"{handled.std() - series.std():+.2f}")
handle_cols[2].metric("N", f"{handled.count()}", f"{handled.count() - series.count():+d}")

st.text_area(
    "Pros and cons of the strategy you chose, and the specific clinical risk if this variable's outliers "
    "are mishandled:",
    key="m1_rigor_handle_notes",
)

st.subheader("2.5 Report them")
st.markdown(
    """
Everything above is invisible to a reader unless you write it down. Transparent reporting is not a
courtesy — it is what makes the result reproducible, and what lets a reviewer tell a cleaning decision
from a result.
"""
)
st.text_area(
    "Write the outlier-handling sentence that would appear in your methods section. Name the rule, the "
    "variables it was applied to, how many records it affected, and what you did with them.",
    key="m1_rigor_reflection",
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — Validation strategy
# ═══════════════════════════════════════════════════════════════════════════
st.header("3. A Validation Strategy You Can Defend")

st.markdown(
    """
**The situation:** your institution has **10,000 CHF admissions from three hospitals over three years**,
recorded on five different EHR configurations, in a diverse, mostly urban population. The model will later
be tested at an outside health system.
"""
)

st.subheader("Task 1 — Splitting")
split_issues = st.multiselect(
    "What makes a simple random split unsafe for *this* dataset? (choose all that apply)",
    [
        "Temporal leakage — future data ends up in the training set",
        "The same patient appears in both train and test sets",
        "Class imbalance differs across hospitals",
        "Site and equipment batch effects are ignored",
        "Demographic subgroups are unevenly represented across the split",
        "Readmission prevalence changed over the three years",
    ],
    key="m1_valid_split_issues",
)
split_strategy = st.radio(
    "Which splitting principle will you commit to?",
    [
        "Temporal split (train on the first two years, test on the last)",
        "Hospital-wise split (hold out one hospital entirely)",
        "Patient-level split stratified by outcome",
        "Hybrid (temporal + hospital + demographic stratification)",
    ],
    key="m1_valid_split_strategy",
)
st.text_area(
    "Which of the issues you selected does your chosen split actually solve — and which does it leave open?",
    key="m1_valid_split_notes",
)

st.subheader("Task 2 — Internal validation")
cv_cols = st.columns([2, 1])
with cv_cols[0]:
    cv_type = st.selectbox(
        "Cross-validation design:",
        [
            "K-fold (random)",
            "K-fold (hospital-stratified)",
            "Leave-one-hospital-out",
            "Time-series (rolling window)",
            "Nested CV (tuning inside, evaluation outside)",
        ],
        key="m1_valid_cv_type",
    )
with cv_cols[1]:
    n_folds = st.slider("Folds:", 3, 10, 5, key="m1_valid_folds")

cv_metrics = st.multiselect(
    "Which metrics will you report per fold? (accuracy alone is not enough on imbalanced outcomes)",
    [
        "AUROC",
        "AUPRC",
        "Sensitivity / recall",
        "Specificity",
        "Calibration (Brier score, reliability curve)",
        "Subgroup disparities",
    ],
    key="m1_valid_metrics",
)
if cv_metrics and "Calibration (Brier score, reliability curve)" not in cv_metrics:
    st.caption(
        "Worth reconsidering: a discharge-planning tool acts on the *probability*, not the ranking. "
        "Discrimination without calibration will not tell you whether 0.7 means 70%."
    )

st.subheader("Task 3 — External validation")
st.text_area(
    "Your model will be evaluated at an outside health system whose coding practices and case mix differ "
    "from yours. What will you hold fixed, what will you allow to be re-fit, and what result would make "
    "you say the model does *not* generalize?",
    key="m1_valid_external",
)

st.subheader("Task 4 — Subgroup performance")
subgroups = st.multiselect(
    "Which subgroups will you report performance for, separately, before anyone deploys this?",
    [
        "Age bands",
        "Sex",
        "Race and ethnicity",
        "Insurance status / payer",
        "Primary language",
        "Hospital site",
        "Prior-admission count",
    ],
    key="m1_valid_subgroups",
)
st.text_area(
    "Pick the subgroup you expect to perform worst and say why — mechanism, not guesswork. What would you "
    "do if you were right?",
    key="m1_valid_subgroup_notes",
)

if split_strategy and cv_type:
    st.info(
        f"**Your stated design:** {split_strategy.split(' (')[0]} · {cv_type} with {n_folds} folds · "
        f"{len(cv_metrics)} reported metric(s) · {len(subgroups)} subgroup(s) audited."
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Part 4 — Carrying it to the team
# ═══════════════════════════════════════════════════════════════════════════
st.header("4. Carrying the Decision to Your Team")

st.markdown(
    """
Your study needs an emergency physician, a cardiologist, a data scientist, a nurse informaticist, and
someone who can speak for patients. The problem is they do not share a vocabulary and words have very
different meanings. Look at a few examples below:
"""
)

with st.expander("Terms this team will use differently without noticing", expanded=False):
    st.markdown(
        """
    Before you write anything, check that these mean the same thing to everyone in the room:

    | Term | Where the confusion comes from |
    | --- | --- |
    | *validation* | statistical out-of-sample testing vs. regulatory/clinical validation |
    | *sensitivity* | true-positive rate vs. "how twitchy the alert is" |
    | *significance* | p < 0.05 vs. "big enough to change what I do" |
    | *model* | the fitted artifact vs. the whole deployed system |
    | *bias* | statistical estimation bias vs. social/structural inequity |
    | *label* | the recorded outcome vs. the clinical truth it stands in for |
    | *drift* | covariate shift vs. "the ward changed its protocol" |
    | *positive* | predicted positive vs. the patient actually being readmitted |
    | *feature* | model input vs. product functionality |
    | *accuracy* | the metric vs. "is it right" |
    """
    )

st.text_area(
    "**The team question.** Who do you need on this team, what will each of them catch that you would "
    "miss, and how will you keep the clinical reasoning honest as the modelling work speeds up?",
    key="m1_team_plan",
)

st.subheader("The communication artifact")

with st.expander("The situation (click to expand)", expanded=True):
    st.markdown(
        """
    **You are Dr. Jordan**, an early-career clinician-researcher. You are mentored by **Dr. Martinez**, a
    senior attending and clinical AI lead.

    Early mentorship meetings were productive — Dr. Martinez helped you integrate risk-prediction models
    into heart-failure workflows. Lately, meetings have been rescheduled or cut short. You feel
    unsupported, with a multicenter protocol deadline and IRB submission approaching.

    Dr. Martinez, for their part, sees you as becoming dependent — waiting for advice rather than
    troubleshooting data and workflow obstacles yourself. Frustration is building on both sides.
    """
    )

st.markdown(
    """
Write the message that requests a meeting and actually improves the situation. It has to do three things
at once: state the problem clearly, show you understand the other side, and propose a concrete change.
This is the same skill as defending your design to an IRB or a sceptical department — the audience is
busy and the ask has to be specific.
"""
)

st.text_area(
    "Your message to Dr. Martinez:",
    height=220,
    key="m1_comm_email",
)

if st.button("Compare with a worked example", key="m1_comm_example_btn"):
    st.info(
        """\
Dear Dr. Martinez,

I hope this message finds you well. I'd like to request a meeting to discuss our current working
relationship and some challenges I've been experiencing. Your insights on integrating AI into our heart
failure protocols have been extremely valuable, and I am grateful for your mentorship.

Recently, I've noticed our meetings have become less frequent and are sometimes cut short. I completely
appreciate your clinical and research obligations, especially as new projects arise. However, with the
upcoming multicenter protocol deadline and IRB submission, I've felt unsure at times how to proceed when
obstacles arise.

I also realize I could be more proactive in troubleshooting workflow bottlenecks before seeking your
direct guidance. Would we be able to set a recurring check-in (even biweekly) and perhaps agree on short
agendas to maximize our time? I'd like to become more independent but still benefit from your targeted
advice during critical moments.

Thank you for considering this. I am eager to find a balance that supports both your schedule and my
professional growth.

Best regards,
Dr. Jordan
"""
    )
    st.caption(
        "Notice what it does: names the specific change (frequency), credits the other side's constraints, "
        "concedes its own contribution to the problem, and asks for one concrete, cheap thing."
    )

st.markdown(
    """
---
**Key takeaways**

- A defensible study states its gap, its question, and its data plan *before* the modelling starts.
- Cleaning decisions are results. Report the rule, the count, and the consequence.
- Splitting, cross-validation, external validation, and subgroup performance are four separate claims —
  a strong answer to one does not cover the others.
- High-stakes, time-pressured teams need explicit expectations more than they need extra meetings.

**Further reading:** [Nature — How to be a good mentee](https://www.nature.com/articles/d41586-020-02927-0) ·
[Effective mentoring in clinical research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4564451/)
"""
)
