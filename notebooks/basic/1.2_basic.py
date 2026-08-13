import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.markdown(
    """
You are a computational biologist in a lab studying how cells respond to drug compounds. You measure
**cell viability** — how many cells are still alive in a plate after treatment — and you want to use AI to
**score viability from images** instead of counting by hand.

A design is only worth as much as the rigor behind it. This subsection moves in four steps:

1. **Write the design brief** — the gap, the question, and the data plan.
2. **Do the rigor work** — detect and handle outliers in a real table, and see what your choice does to the numbers.
3. **Commit to a validation strategy** — splitting, cross-validation, external validation, subgroup performance.
4. **Carry the decision to your team** — one professional message that a busy senior colleague will actually read.

**Resources.** The two collections below are the kind of imaging data a study like this runs on, and the
two tools are what you would build it with — worth opening once now so the data plan in Part 1 is concrete
rather than hypothetical:
[Cell Painting Gallery](https://broadinstitute.github.io/cellpainting-gallery/) and
[Allen Brain Atlas](https://portal.brain-map.org/) (public, openly downloadable image sets) ·
[PyTorch](https://pytorch.org/) and [scikit-learn](https://scikit-learn.org/stable/) (where the analysis
would actually run).
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
    "be most relevant as you design an AI study to score cell viability from images after drug treatment.",
    key="m1_design_gap",
)
st.text_area(
    "**Question.** State one primary research question that closes that gap, using SMART criteria "
    "(specific, measurable, achievable, relevant, time-bound).",
    key="m1_design_question",
)

st.subheader("1.2 The data plan")
channels = st.multiselect(
    "**Elements.** Which staining channels will your model see?",
    [
        "Nuclear stain (total cell count)",
        "Live-cell stain (e.g. calcein-AM)",
        "Dead-cell stain (e.g. propidium iodide)",
        "Mitochondrial membrane potential",
        "Brightfield / phase contrast",
        "Cytoskeleton",
    ],
    key="m1_design_elements",
)
st.text_area(
    "**Cohort.** Give your inclusion and exclusion criteria for images in two or three lines "
    "(cell line, focus quality, treatment and dose, plate position).",
    key="m1_design_cohort",
)
st.text_area(
    "**Missingness and bias.** How will you handle failed wells and missing channels, and which "
    "acquisition bias — batch, plate edge, illumination, cell line — are you most worried about carrying "
    "into the model?",
    key="m1_design_missing",
)
st.text_area(
    "**Preprocessing.** Name the transformations you will apply — illumination correction, intensity "
    "normalization, segmentation, and at least one biologically meaningful derived feature "
    "(e.g., the live-to-total cell ratio).",
    key="m1_design_prepro",
)

if channels:
    st.caption(f"Your model will see: {', '.join(channels)}.")
    if len(channels) == 1:
        st.caption(
            "Note: a single channel makes interpretation easier, but viability is a *ratio* — without a "
            "total-count channel to divide by, a drop in live cells and a drop in seeding look identical."
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
    | **Gap** | It names a limitation you could show evidence for — a cell line the method was never tested on, a dose range it cannot resolve, a batch effect it ignores — not "more research is needed". |
    | **Question** | A reader can tell what you will measure, in what system, and by when. If it has no measurable outcome, it is a topic, not a question. |
    | **Elements** | The channels you ticked can actually produce a viability number. Live and dead counts without a total count give you a difference, not a fraction. |
    | **Cohort** | Someone else could apply your criteria to the same plates and keep the same images. |
    | **Missingness and bias** | You named a handling rule *and* a named bias — and said which direction the bias would push the model, not just that it exists. |
    | **Preprocessing** | Each transformation is one you could compute from the channels you selected above. |

    The most common failure is a question the data plan cannot answer. If your question mentions
    something your selected channels never record, one of the two has to change.
    """
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — The rigor lab (outliers)
# ═══════════════════════════════════════════════════════════════════════════
st.header("2. Rigor Lab: Outliers in Your Replicate Counts")

st.markdown(
    """
Your brief promised to handle failed wells and extreme values. This is where you actually do it.

Below is a **simulated cell-viability table** of 30 wells. Each well was treated, then counted three
times — once in each of three replicate experiments — so every row holds the same measurement made three
independent ways:

- `replicate_1` — live cells counted in the first replicate experiment
- `replicate_2` — the same well, second replicate
- `replicate_3` — the same well, third replicate

Replicates are where counting errors show up. A well that reads 800, 780, and 41 was not killed by the
drug in one experiment out of three — one replicate did not count correctly.
"""
)


@st.cache_data
def load_viability_replicates():
    rng = np.random.default_rng(42)
    n = 30
    # A well's true viability is shared across replicates; the per-replicate noise is counting noise.
    true_count = rng.normal(760, 120, n)
    return pd.DataFrame(
        {
            "well_id": [f"W{i:02d}" for i in range(1, n + 1)],
            # replicate 2 saturated on one well; replicate 3 miscounted two wells (one empty, one doubled)
            "replicate_1": np.round(true_count + rng.normal(0, 35, n)),
            "replicate_2": np.round(np.append((true_count + rng.normal(0, 35, n))[:-1], [2400])),
            "replicate_3": np.round(
                np.append((true_count + rng.normal(0, 35, n))[:-2], [41, 1830])
            ),
        }
    )


df = load_viability_replicates()
VARIABLES = ["replicate_1", "replicate_2", "replicate_3"]

st.dataframe(df, width="stretch")


def iqr_bounds(column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr, q1, q3, iqr


st.subheader("2.1 See them")
st.markdown(
    "A boxplot makes an extreme value obvious. All three replicates are plotted on the same axis, so a "
    "count that only one replicate produced has nothing to hide behind."
)

long_counts = df.melt(
    id_vars="well_id", value_vars=VARIABLES, var_name="Replicate", value_name="Live cell count"
)
fig = px.box(
    long_counts, x="Replicate", y="Live cell count", points="all", hover_data=["well_id"]
)
fig.update_layout(
    height=340,
    xaxis_title="",
    margin=dict(l=40, r=20, t=25, b=35),
)
st.plotly_chart(fig, width="stretch")

with st.expander("View chart data as text (accessible alternative)"):
    st.markdown("**Five-number summary of each replicate**")
    st.dataframe(df[VARIABLES].describe().T, width="stretch")
    st.markdown("**The five counts furthest from their own replicate's median**")
    st.dataframe(
        long_counts.assign(
            distance_from_median=lambda d: (
                d["Live cell count"] - d.groupby("Replicate")["Live cell count"].transform("median")
            ).abs()
        )
        .sort_values("distance_from_median", ascending=False)
        .head(5)[["well_id", "Replicate", "Live cell count"]],
        width="stretch",
        hide_index=True,
    )

st.text_area(
    "Which points look like outliers, which wells are they, and which replicate did each come from?",
    key="m1_rigor_visual_notes",
)

st.subheader("2.2 Measure them")
st.markdown(
    "The 1.5×IQR rule flags any value above Q3 + 1.5×IQR or below Q1 − 1.5×IQR. It is a convention, not "
    "a law — but it is a convention you can write down in a methods section."
)

sel_stat = st.selectbox("Replicate for threshold calculation:", VARIABLES, key="m1_rigor_calc_var")
lower, upper, q1, q3, iqr = iqr_bounds(sel_stat)

bound_cols = st.columns(3)
bound_cols[0].metric("IQR", f"{iqr:.2f}", help=f"Q1 = {q1:.2f}, Q3 = {q3:.2f}")
bound_cols[1].metric("Lower bound", f"{lower:.2f}")
bound_cols[2].metric("Upper bound", f"{upper:.2f}")

outlier_mask = (df[sel_stat] < lower) | (df[sel_stat] > upper)
st.markdown("**Rows flagged by the IQR rule:**")
if outlier_mask.any():
    st.dataframe(df[outlier_mask], width="stretch")
else:
    # replicate_1 is clean. An empty table with no caption reads as a broken page, so say plainly that
    # a replicate with nothing to flag is a result rather than a missing one.
    st.success(
        f"Nothing flagged. Every count in **{sel_stat}** falls inside the bounds above — that replicate "
        "counted cleanly. A replicate with no outliers is a finding, not a failed check. Switch to "
        "another replicate to see what a failed count looks like."
    )
st.text_area(
    "Do the flagged counts look like a replicate that failed to count, or like wells that genuinely "
    "responded to the drug? Compare each flagged well against its other two replicates before you answer — "
    "and note that your answer changes what you are allowed to do next.",
    key="m1_rigor_flagged_notes",
)

st.subheader("2.3 See what they cost you")
sel_compare = st.selectbox("Replicate for comparison:", VARIABLES, key="m1_rigor_compare_var")
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

- **Remove** — honest about uncertainty, but throws away real wells and shrinks your dataset.
- **Winsorize** — keeps every well, at the price of a count that was never observed.
- **Impute with median** — keeps the well and the sample size, and erases the signal that made it unusual.
"""
)

sel_handle = st.selectbox("Replicate for handling strategies:", VARIABLES, key="m1_rigor_handle_var")
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
    "Pros and cons of the strategy you chose, and where it would be the wrong choice:",
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
**The situation:** your dataset is **10,000 cell counts acquired over the year**, across three plate
batches, two microscopes, and four cell lines. The model will later be tested on images from a
collaborating lab using a different imaging platform to measure cell viability.
"""
)

st.subheader("Task 1 — Splitting")
split_issues = st.multiselect(
    "What makes a simple random split unsafe for *this* dataset? (choose all that apply)",
    [
        "Temporal leakage — later acquisitions end up in the training set",
        "Images of the same well or field appear in both train and test",
        "Phenotype class balance differs across batches",
        "Microscope and illumination batch effects are ignored",
        "Some cell lines are represented on only one plate",
        "Rare phenotypes may be absent from the test set entirely",
    ],
    key="m1_valid_split_issues",
)
split_strategy = st.radio(
    "Which splitting principle will you commit to?",
    [
        "Temporal split (train on earlier batches, test on the latest)",
        "Leave-one-batch-out (hold out an entire plate batch)",
        "Well-level split stratified by phenotype",
        "Hybrid (batch + cell line + phenotype stratification)",
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
            "K-fold (stratified by phenotype)",
            "K-fold (grouped by cell line)",
            "Leave-one-batch-out",
            "Nested CV (tuning inside, evaluation outside)",
        ],
        key="m1_valid_cv_type",
    )
with cv_cols[1]:
    n_folds = st.slider("Folds:", 3, 10, 5, key="m1_valid_folds")

cv_metrics = st.multiselect(
    "Which metrics will you report per fold? (accuracy alone is not enough when phenotypes are rare)",
    [
        "Accuracy",
        "Macro F1",
        "Precision / recall per class",
        "AUROC",
        "Confusion matrix",
        "Biological enrichment (GO, pathways)",
        "Cluster purity / silhouette",
    ],
    key="m1_valid_metrics",
)
if cv_metrics and "Accuracy" in cv_metrics and len(cv_metrics) == 1:
    st.caption(
        "Worth reconsidering: with a rare phenotype, a model that never predicts it can still be 97% "
        "accurate. You need at least one per-class metric."
    )

st.subheader("Task 3 — External validation")
st.text_area(
    "Your model will be evaluated on images from a collaborating lab using a different microscope and "
    "staining protocol. What will you hold fixed, what will you allow to be re-fit, and what result would "
    "make you say the model does *not* generalize?",
    key="m1_valid_external",
)

st.subheader("Task 4 — Subgroup performance")
subgroups = st.multiselect(
    "Which strata will you report performance for, separately, before claiming the model works?",
    [
        "Cell line",
        "Plate batch",
        "Microscope / imaging platform",
        "Compound class",
        "Dose level",
        "Plate position (edge vs. interior wells)",
        "Rare vs. common phenotypes",
    ],
    key="m1_valid_subgroups",
)
st.text_area(
    "Pick the stratum you expect to perform worst and say why — mechanism, not guesswork. What would you "
    "do if you were right?",
    key="m1_valid_subgroup_notes",
)

if split_strategy and cv_type:
    st.info(
        f"**Your stated design:** {split_strategy.split(' (')[0]} · {cv_type} with {n_folds} folds · "
        f"{len(cv_metrics)} reported metric(s) · {len(subgroups)} stratum/strata audited."
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Part 4 — Carrying it to the team
# ═══════════════════════════════════════════════════════════════════════════
st.header("4. Carrying the Decision to Your Team")

st.markdown(
    """
Your study needs a cell biologist, a microscopist, a data scientist, a software engineer, and someone who
knows the compound library. The problem is they do not share a vocabulary and words have very different
meanings. Look at a few examples below:
"""
)

with st.expander("Terms this team will use differently without noticing", expanded=False):
    st.markdown(
        """
    Before you write anything, check that these mean the same thing to everyone in the room:

    | Term | Where the confusion comes from |
    | --- | --- |
    | *validation* | statistical out-of-sample testing vs. wet-lab confirmation |
    | *replicate* | technical replicate vs. biological replicate |
    | *significance* | p < 0.05 vs. "big enough to be a real phenotype" |
    | *model* | the fitted classifier vs. the model organism or system |
    | *bias* | statistical estimation bias vs. batch/acquisition bias |
    | *label* | the annotated class vs. the underlying biology it stands for |
    | *feature* | model input vs. a visible cellular structure |
    | *control* | negative control well vs. experimental control condition |
    | *normalization* | per-image intensity scaling vs. per-plate statistical normalization |
    | *accuracy* | the metric vs. "is it right" |
    """
    )

st.text_area(
    "**The team question.** Who do you need on this team, what will each of them catch that you would "
    "miss, and how will you keep the biology honest as the modelling work speeds up?",
    key="m1_team_plan",
)

st.subheader("The communication artifact")

with st.expander("The situation (click to expand)", expanded=True):
    st.markdown(
        """
    **You are Dr. Witmer**, an early-career research faculty member, mentored by **Dr. Antone**, a senior
    faculty member.

    Early meetings were productive, but over time Dr. Antone has become less available — often
    rescheduling or cutting meetings short. You feel unsupported, particularly while preparing an upcoming
    grant application.

    Dr. Antone, in turn, sees you as overly reliant and not proactive in solving problems independently.
    Tensions are rising and both of you are frustrated.
    """
    )

st.markdown(
    """
Write the message that requests a meeting and actually improves the situation. It has to do three things
at once: state the problem clearly, show you understand the other side, and propose a concrete change.
This is the same skill as defending your design to a study section or a sceptical collaborator — the
audience is busy and the ask has to be specific.
"""
)

st.text_area(
    "Your message to Dr. Antone:",
    height=220,
    key="m1_comm_email",
)

if st.button("Compare with a worked example", key="m1_comm_example_btn"):
    st.info(
        """\
Dear Dr. Antone,

I hope you're doing well. I'd like to request a meeting to discuss our working relationship and the
challenges that I have been facing recently. I truly value your expertise, and I have learned a lot from
you.

Over the past few months, I've noticed that our meetings have been less frequent, with some rescheduled or
cut short. I understand that you have many demands on your time, but I've felt unsupported, especially as
I prepare for the upcoming grant application. However, I recognize that I may have been overly reliant on
you, and I want to be more proactive in problem-solving going forward. Maybe we could schedule regular
check-ins, with clear agendas to make the most of our time together. Mainly, I want to find a way to be
more self-reliant while still benefiting from your mentorship.

I look forward to hearing your thoughts on this.

Best,
Dr. Witmer
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
- Open expectations prevent small misalignments from becoming ruptures.

**Further reading:** [Nature — How to be a good mentee](https://www.nature.com/articles/d41586-020-02927-0) ·
[Science — How to make the most of mentoring](https://www.science.org/content/article/how-make-most-mentoring)
"""
)
