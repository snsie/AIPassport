import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import lime
import lime.lime_tabular
import aipassport_config as cfg
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# ── Track-specific framing (this file is the basic science track) ───────────
DEFAULT_DATASET_INDEX = 1  # basic science track opens on the synthetic cohort

st.markdown(
    """
Subsection 4.1 built a model. Now we have to evaluate if we trust the model. Let's consider these four
questions sequentially:

1. **Where does it start memorizing?** Compare a model that is too complex with one that is too simple.
2. **Which validation strategy earns the number you report?**
3. **Does it work equally well for everyone?** Aggregate accuracy hides subgroup failure.
4. **Why did it say that?** Globally with SHAP, for one individual with LIME, and by hand with a what-if
   simulator.

All four share **one dataset, one split, and one scaler** — chosen immediately below.
"""
)


def render_plotly_chart(fig, height=520):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch", config={"responsive": True})


@st.cache_data
def load_eicu_subset():
    """Bundled eICU demo extract. No network access: the CSV ships with the repository."""
    df = pd.read_csv("assets/datasets/csv/eicu_demo.csv")
    if "weight" in df.columns and "weight_admission" not in df.columns:
        df = df.rename(columns={"weight": "weight_admission"})
    feats = ["age", "lab_glucose", "lab_creatinine", "lab_potassium"]
    for col in feats:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feats] = df[feats].fillna(df[feats].mean())
    df = df.dropna(subset=["in_hospital_mortality"])
    sample = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)
    return sample, feats, "in_hospital_mortality"


@st.cache_data
def load_synthetic_cohort():
    """A synthetic, well-separated dataset: useful when you need a known ground truth."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=2,
        n_redundant=1,
        n_clusters_per_class=1,
        flip_y=0.1,
        random_state=42,
    )
    features = ["gene_x_expression", "protein_y_level", "culture_ph", "temperature_c"]
    df = pd.DataFrame(X, columns=features)
    df["gene_x_expression"] = (df["gene_x_expression"] * 10) + 50
    df["protein_y_level"] = (df["protein_y_level"] * 5) + 20
    df["culture_ph"] = (df["culture_ph"] * 0.5) + 7.4
    df["temperature_c"] = (df["temperature_c"] * 1.5) + 37.0
    df["cellular_apoptosis"] = y
    return df, features, "cellular_apoptosis"


DATASETS = {
    "eICU cohort (real ICU admissions, in-hospital mortality)": load_eicu_subset,
    "Synthetic cohort (simulated cell measurements, predicting apoptosis, known ground truth)": load_synthetic_cohort,
}

dataset_choice = st.selectbox(
    "Dataset for all four activities:",
    list(DATASETS),
    index=DEFAULT_DATASET_INDEX,
    help="This is a dataset selector, not a track selector. The real cohort is messier and more "
    "realistic; the synthetic one has a known ground truth, which makes overfitting easier to see.",
    key="m4_eval_dataset",
)

df, FEATURES, TARGET = DATASETS[dataset_choice]()

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# The one model activities 3 and 4 share: activity 3 audits it by subgroup, activity 4 explains it. It is
# fitted here rather than inside activity 3 because only the selected activity's body executes, and
# activity 4's explanations have to describe the same fitted model the audit just scored.
fair_model = LogisticRegression(solver="lbfgs", max_iter=1000)
fair_model.fit(X_train, y_train)

st.caption(
    f"{len(df)} records · {len(FEATURES)} features · target `{TARGET}` · "
    f"{len(X_train)} train / {len(X_test)} test (stratified, random_state=42)."
)

ACTIVITIES = [
    "1. Overfitting and Tuning",
    "2. Validation Strategies",
    "3. Subgroup Fairness",
    "4. Explaining Predictions",
]
# A keyed segmented_control rather than st.tabs: tab selection lives in the browser and is
# lost whenever a widget inside a tab triggers a rerun, which is what sent learners back to
# the first activity mid-edit. This selection is in session_state, so it survives.
activity = st.segmented_control(
    "Activity",
    ACTIVITIES,
    default=ACTIVITIES[0],
    key="m4_eval_activity",
    required=True,
)
# ═══════════════════════════════════════════════════════════════════════════
# 1 — Overfitting and tuning
# ═══════════════════════════════════════════════════════════════════════════
if activity == ACTIVITIES[0]:
    st.header("Activity 1: Where the Model Starts Memorizing")
    st.markdown(
        """
    A k-nearest-neighbours model has exactly one knob, which makes it the clearest possible illustration.
    **Low k** draws a boundary around every individual point. **High k** draws one broad region and ignores
    local structure. Neither generalizes; the answer is in between, and you find it by measurement.
    """
    )

    st.subheader("Decision boundaries (first two features)")
    c1, c2 = st.columns(2)
    k_low = c1.number_input(
        "Complex model (k)", 1, 50, 1, help="A low k is highly sensitive to noise.", key="m4_eval_k_low"
    )
    k_high = c2.number_input(
        "Simple model (k)", 1, 50, 15, help="A high k smooths the boundary out.", key="m4_eval_k_high"
    )

    def boundary_data(k_val):
        X_2d = X_train_s[:, :2]
        knn = KNeighborsClassifier(n_neighbors=k_val).fit(X_2d, y_train)
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        return X_2d, xx, yy, Z

    fig_b = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Overfitting (k={k_low})", f"Underfitting (k={k_high})"),
        horizontal_spacing=0.08,
    )
    for col, k_val in ((1, k_low), (2, k_high)):
        x_2d, xx, yy, z = boundary_data(k_val)
        fig_b.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=z,
                colorscale="Cividis",
                opacity=0.75,
                showscale=False,
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        fig_b.add_trace(
            go.Scatter(
                x=x_2d[:, 0],
                y=x_2d[:, 1],
                mode="markers",
                marker=dict(
                    color=y_train, colorscale="Cividis", size=7, line=dict(color="white", width=1)
                ),
                showlegend=False,
                hovertemplate=(
                    f"{FEATURES[0]}: %{{x:.2f}}<br>{FEATURES[1]}: %{{y:.2f}}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        fig_b.update_xaxes(title_text=f"{FEATURES[0]} (scaled)", row=1, col=col)
        fig_b.update_yaxes(title_text=f"{FEATURES[1]} (scaled)", row=1, col=col)
    render_plotly_chart(fig_b, height=560)
    st.caption(
        "Colourblind-accessible contour plot of the two models' decision boundaries. Yellow regions predict "
        "one class, dark blue the other; white-outlined dots are individual records. The left panel's "
        "islands are memorized points, not structure."
    )

    st.subheader("The accuracy curve — where the two lines part")
    st.markdown(
        "Now measure it. Train accuracy and test accuracy agree while the model is learning general rules, "
        "and separate the moment it starts memorizing. That divergence is the signature of overfitting."
    )
    k_max = st.slider(
        "Maximum k to test", 5, 50, 20, help="Extend this to see the model slide into underfitting.",
        key="m4_eval_kmax",
    )

    ks = list(range(1, k_max + 1))
    train_acc, test_acc = [], []
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k).fit(X_train_s, y_train)
        train_acc.append(accuracy_score(y_train, knn.predict(X_train_s)))
        test_acc.append(accuracy_score(y_test, knn.predict(X_test_s)))

    fig_acc = go.Figure()
    fig_acc.add_trace(
        go.Scatter(x=ks, y=train_acc, mode="lines+markers", name="Train accuracy",
                   line=dict(color=cfg.CHART_PRIMARY))
    )
    fig_acc.add_trace(
        go.Scatter(x=ks, y=test_acc, mode="lines+markers", name="Test accuracy",
                   line=dict(color=cfg.CHART_SECONDARY))
    )
    best_k = ks[int(np.argmax(test_acc))]
    fig_acc.add_vline(
        x=best_k, line_dash="dash", line_color=cfg.DANGER, annotation_text=f"best test k={best_k}"
    )
    fig_acc.update_xaxes(title_text="Number of neighbours (k)")
    fig_acc.update_yaxes(title_text="Accuracy", range=[0, 1.05])
    render_plotly_chart(fig_acc, height=470)

    with st.expander("View chart data as text (accessible alternative)"):
        st.dataframe(
            pd.DataFrame(
                {"k": ks, "Train accuracy": train_acc, "Test accuracy": test_acc}
            ).round(3),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "Dark blue is training accuracy, yellow is test accuracy. At k=1 training accuracy is perfect and "
        "meaningless — the nearest neighbour of a training point is itself."
    )

    gap = train_acc[0] - test_acc[0]
    st.metric("Train-test gap at k=1", f"{gap:.3f}", help="The size of the memorization effect.")

    with st.expander("Reveal concept summary"):
        st.write(
            f"The best test accuracy on this split occurs at k={best_k}. Choose the hyperparameter just "
            "before the curves diverge substantially. A large gap between high training accuracy and low "
            "test accuracy is the mathematical signature of overfitting — and note that you can only see it "
            "because the test set was held back."
        )

# ═══════════════════════════════════════════════════════════════════════════
# 2 — Validation strategies
# ═══════════════════════════════════════════════════════════════════════════
if activity == ACTIVITIES[1]:
    st.header("Activity 2: Which Validation Strategy Earns Your Number")
    st.markdown(
        "The accuracy in activity 1 came from one split. Change the split and it changes. Cross-validation shows "
        "you the whole distribution, and *how* you fold matters as much as how many times."
    )

    col_cv1, col_cv2 = st.columns(2)
    n_f = col_cv1.slider(
        "Number of folds", 2, 10, 5, help="How many subsets the data is split into.", key="m4_eval_folds"
    )
    k_cv = col_cv2.slider(
        "Model complexity (k)", 1, 20, 5, help="Neighbour count for the model being validated.",
        key="m4_eval_kcv",
    )

    knn_cv = KNeighborsClassifier(n_neighbors=k_cv)
    X_s = StandardScaler().fit_transform(X)
    kf_scores = cross_val_score(knn_cv, X_s, y, cv=KFold(n_splits=n_f, shuffle=True, random_state=42))

    fig_cv = go.Figure()
    fig_cv.add_trace(
        go.Box(
            x=kf_scores,
            name=f"{n_f} folds",
            boxpoints="all",
            jitter=0.35,
            pointpos=0,
            marker_color=cfg.CHART_SECONDARY,
            line_color=cfg.CHART_PRIMARY,
        )
    )
    fig_cv.update_xaxes(title_text="Accuracy", range=[0, 1.05])
    fig_cv.update_layout(title=f"Accuracy distribution across {n_f} folds")
    render_plotly_chart(fig_cv, height=400)

    with st.expander("View chart data as text (accessible alternative)"):
        st.dataframe(
            pd.DataFrame(
                {"Fold": range(1, len(kf_scores) + 1), "Accuracy": kf_scores}
            ).round(3),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "Each dot is one fold. A wide box means the model's performance depends heavily on which records "
        "happened to land in the test fold — so a single split would have been a coin toss."
    )

    st.subheader("K-Fold vs. Stratified K-Fold")
    st.markdown(
        "Plain K-Fold slices the data without regard to the outcome. With an imbalanced target, a fold can "
        "end up with almost none of the minority class, and its score becomes noise."
    )

    skf_scores = cross_val_score(
        knn_cv, X_s, y, cv=StratifiedKFold(n_splits=n_f, shuffle=True, random_state=42)
    )

    methods = ["K-Fold", "Stratified K-Fold"]
    means = [kf_scores.mean(), skf_scores.mean()]
    stds = [kf_scores.std(), skf_scores.std()]

    fig_bar = go.Figure(
        go.Bar(
            x=methods,
            y=means,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color=[cfg.CHART_PRIMARY, cfg.CHART_SECONDARY],
            text=[f"{m:.3f}" for m in means],
        )
    )
    fig_bar.update_yaxes(title_text="Mean accuracy", range=[0, 1.1])
    render_plotly_chart(fig_bar, height=430)

    with st.expander("View chart data as text (accessible alternative)"):
        st.dataframe(
            pd.DataFrame(
                {"Method": methods, "Mean accuracy": means, "Std. dev. across folds": stds}
            ).round(4),
            width="stretch",
            hide_index=True,
        )

    strat_cols = st.columns(2)
    strat_cols[0].metric("K-Fold std. dev.", f"{kf_scores.std():.4f}")
    strat_cols[1].metric(
        "Stratified std. dev.",
        f"{skf_scores.std():.4f}",
        delta=f"{skf_scores.std() - kf_scores.std():+.4f}",
        delta_color="inverse",
    )

    minority_share = float(min(y.mean(), 1 - y.mean()))
    st.caption(
        f"The minority class is {minority_share:.1%} of this dataset. Stratified K-Fold guarantees that "
        f"ratio in every fold, which is why it is the default choice for biomedical data. "
        "**Leave-one-out CV** is the limiting case — one fold per record. It is nearly unbiased and "
        f"prohibitively expensive: on these {len(df)} records it would fit {len(df)} models to answer the "
        "same question these folds already answered, which is why it is described here rather than run."
    )

# ═══════════════════════════════════════════════════════════════════════════
# 3 — Subgroup fairness
# ═══════════════════════════════════════════════════════════════════════════
if activity == ACTIVITIES[2]:
    st.header("Activity 3: Does It Work Equally Well for Everyone?")
    st.markdown(
        "One aggregate number can hide a model that works well for the majority and fails for a subgroup. "
        "The only way to find out is to split the test set and score each stratum separately."
    )

    overall_cols = st.columns(3)
    overall_pred = fair_model.predict(X_test)
    overall_cols[0].metric("Overall accuracy", f"{accuracy_score(y_test, overall_pred):.3f}")
    overall_cols[1].metric(
        "Overall sensitivity", f"{recall_score(y_test, overall_pred, zero_division=0):.3f}"
    )
    tn_o, fp_o, _, _ = confusion_matrix(y_test, overall_pred, labels=[0, 1]).ravel()
    overall_cols[2].metric(
        "Overall specificity", f"{tn_o / (tn_o + fp_o) if (tn_o + fp_o) else 0:.3f}"
    )

    st.subheader("Now split it")
    subgroup_feature = st.selectbox(
        "Stratify the test set by:",
        FEATURES,
        help="Any feature can define a subgroup. Pick the one whose subgroups you would actually have to "
        "answer for.",
        key="m4_eval_subgroup_feature",
    )

    lo = float(df[subgroup_feature].min())
    hi = float(df[subgroup_feature].max())
    cut_low, cut_high = st.slider(
        f"Boundaries for the three {subgroup_feature} strata:",
        lo,
        hi,
        (lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3),
        key="m4_eval_subgroup_cuts",
    )

    X_test_grouped = X_test.copy()
    X_test_grouped["Stratum"] = pd.cut(
        X_test_grouped[subgroup_feature],
        bins=[-np.inf, cut_low, cut_high, np.inf],
        labels=["Low", "Middle", "High"],
    )

    metrics_list = []
    for group in ["Low", "Middle", "High"]:
        idx = X_test_grouped["Stratum"] == group
        if not idx.any():
            continue
        g_pred = fair_model.predict(X_test[idx.values])
        g_true = y_test[idx.values]
        tn, fp, fn, tp = confusion_matrix(g_true, g_pred, labels=[0, 1]).ravel()
        metrics_list.append(
            {
                f"{subgroup_feature} stratum": group,
                "Samples": int(len(g_true)),
                "Positives": int(g_true.sum()),
                "Accuracy": accuracy_score(g_true, g_pred),
                "Sensitivity": recall_score(g_true, g_pred, zero_division=0),
                "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            }
        )

    fair_df = pd.DataFrame(metrics_list)

    col_a, col_b = st.columns([1, 1.5])
    col_a.dataframe(fair_df, width="stretch")

    fig_fair, ax_fair = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=fair_df.melt(
            id_vars=f"{subgroup_feature} stratum",
            value_vars=["Accuracy", "Sensitivity", "Specificity"],
        ),
        x=f"{subgroup_feature} stratum",
        y="value",
        hue="variable",
        ax=ax_fair,
        palette="viridis",
    )
    ax_fair.set_ylim(0, 1.1)
    ax_fair.set_ylabel("Metric score")
    col_b.pyplot(fig_fair)
    plt.close(fig_fair)

    MIN_POSITIVES = 5
    reliable = fair_df[fair_df["Positives"] >= MIN_POSITIVES] if not fair_df.empty else fair_df
    thin = fair_df[fair_df["Positives"] < MIN_POSITIVES] if not fair_df.empty else fair_df

    if not thin.empty:
        st.warning(
            "**Too thin to judge:** "
            + ", ".join(
                f"the {row[f'{subgroup_feature} stratum']} stratum has only "
                f"{int(row['Positives'])} positive case(s)"
                for _, row in thin.iterrows()
            )
            + f". Sensitivity on fewer than {MIN_POSITIVES} positives is noise, not a disparity — a single "
            "record flips it from 0.00 to 1.00. Report it as unestimable rather than as a finding."
        )

    if len(reliable) >= 2:
        worst = reliable.loc[reliable["Sensitivity"].idxmin()]
        spread = reliable["Sensitivity"].max() - reliable["Sensitivity"].min()
        st.metric(
            "Sensitivity spread across estimable strata",
            f"{spread:.3f}",
            help=f"Computed only over strata with at least {MIN_POSITIVES} positive cases.",
        )
        st.markdown(
            f"The **{worst[f'{subgroup_feature} stratum']}** stratum has the lowest sensitivity "
            f"({worst['Sensitivity']:.3f}) on {int(worst['Samples'])} records. Move the boundaries and watch "
            "the spread change — which should tell you something uncomfortable: **how unfair the model looks "
            "depends partly on where you draw the lines.** That is why the strata have to be chosen for "
            "biological reasons and stated in advance, not discovered afterwards."
        )
    elif not fair_df.empty:
        st.info(
            "Fewer than two strata have enough positive cases to compare. Widen the middle band, or accept "
            "that this test set cannot answer the fairness question for this variable — which is itself a "
            "reportable finding."
        )

# ═══════════════════════════════════════════════════════════════════════════
# 4 — Explaining predictions
# ═══════════════════════════════════════════════════════════════════════════
if activity == ACTIVITIES[3]:
    st.header("Activity 4: Why Did It Say That?")
    st.markdown(
        "Three complementary answers to the same question, at three different scales. All use the logistic "
        "regression from activity 3, so the explanations describe the model you just audited."
    )

    st.subheader("Globally: SHAP")
    st.markdown(
        "SHAP attributes each prediction to its features and lets you aggregate across the whole test set. "
        "It answers: *which features drive this model overall, and in which direction?*"
    )

    max_display = st.slider(
        "Features to display", 1, len(FEATURES), min(5, len(FEATURES)), key="m4_eval_shap_n"
    )

    with st.spinner("Computing SHAP values..."):
        explainer = shap.Explainer(fair_model, X_test)
        shap_values = explainer(X_test)

    fig_shap = plt.figure()
    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
    st.pyplot(plt.gcf())
    plt.close(fig_shap)

    with st.expander("View chart data as text (accessible alternative)"):
        st.markdown(
            "Mean absolute SHAP value per feature — how much that feature moves a prediction on "
            "average, regardless of direction. This is the ranking the plot encodes vertically."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": list(X_test.columns),
                    "Mean |SHAP|": np.abs(shap_values.values).mean(axis=0),
                }
            )
            .sort_values("Mean |SHAP|", ascending=False)
            .head(max_display)
            .round(4),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Each dot is one record. Position on the x-axis is that feature's contribution to that record's "
        "prediction; colour is the feature's value. A feature whose dots spread far from zero matters; one "
        "whose dots cluster at zero does not, however biologically important you expected it to be."
    )

    st.divider()
    st.subheader("Locally: LIME")
    st.markdown(
        "A global summary does not tell a specific person why *their* prediction came out the way it did. "
        "LIME fits a simple local model around one record and reports what moved that single decision."
    )

    record_idx = st.number_input(
        "Record index (from the test set)", 0, len(X_test) - 1, 0, key="m4_eval_lime_idx"
    )

    with st.spinner("Explaining this record..."):
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=list(X_train.columns),
            class_names=[f"{TARGET} = 0", f"{TARGET} = 1"],
            mode="classification",
        )
        exp = lime_explainer.explain_instance(
            X_test.iloc[record_idx].values,
            fair_model.predict_proba,
            num_features=min(5, len(FEATURES)),
        )

    lime_cols = st.columns([1.4, 1])
    with lime_cols[0]:
        fig_lime = exp.as_pyplot_figure()
        st.pyplot(fig_lime)
        plt.close(fig_lime)

        with st.expander("View chart data as text (accessible alternative)"):
            st.markdown(
                "Each row is one condition LIME found in this record. A positive weight pushed the "
                "prediction towards class 1; a negative weight pushed it towards class 0."
            )
            st.dataframe(
                pd.DataFrame(exp.as_list(), columns=["Condition", "Weight"]).round(4),
                width="stretch",
                hide_index=True,
            )
    with lime_cols[1]:
        st.markdown(f"**Record #{record_idx}**")
        st.dataframe(X_test.iloc[[record_idx]].T.rename(columns={X_test.index[record_idx]: "Value"}))
        st.markdown(f"True label: **{int(y_test.iloc[record_idx])}**")
        st.markdown(
            f"Predicted probability: **{fair_model.predict_proba(X_test.iloc[[record_idx]])[0][1]:.3f}**"
        )

    st.divider()
    st.subheader("By hand: the what-if simulator")
    st.markdown(
        "The last check is counterfactual. Build a profile, then change one value and see whether the model "
        "moves the way domain knowledge says it should. A model that responds in the wrong direction is "
        "telling you something the accuracy score never will."
    )

    profile = {}
    sim_cols = st.columns(min(4, len(FEATURES)))
    for i, col in enumerate(FEATURES):
        with sim_cols[i % len(sim_cols)]:
            profile[col] = st.slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean()),
                key=f"m4_eval_whatif_{col}",
            )

    input_df = pd.DataFrame([profile])[FEATURES]
    prediction = fair_model.predict(input_df)[0]
    prob = float(fair_model.predict_proba(input_df)[0][1])

    label = f"Positive ({TARGET} = 1)" if prediction == 1 else f"Negative ({TARGET} = 0)"
    if prediction == 1:
        st.error(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")
    st.progress(prob, text=f"Predicted probability: {prob:.1%}")

st.markdown(
    """
---
**Key takeaways**

- Overfitting is measurable: it is the gap between training and held-out performance.
- Stratified K-Fold is the default for biomedical data because it protects the minority class in every fold.
- Aggregate performance is not performance. Report it by subgroup, with the strata chosen in advance.
- SHAP explains the model, LIME explains one prediction, and a counterfactual check tests whether either
  agrees with what you know. A model can be accurate and still be reasoning wrongly.

**Resources:** [SHAP](https://shap.readthedocs.io/) · [LIME](https://github.com/marcotcr/lime) ·
[scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
"""
)
