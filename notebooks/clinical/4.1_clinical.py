import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer, recall_score, confusion_matrix
import aipassport_config as cfg

# ── Track-specific framing (this file is the clinical track) ────────────────
OUTCOME_FRAMING = (
    "You are on a clinical analytics team. The outcome is a **diabetes diagnosis**, and the eight inputs "
    "are the kind of measurements a clinic already has on file. The question is not just whether a model "
    "can predict it — it is whether you can explain the prediction to the person it is about."
)
POSITIVE_LABEL = "Diagnosed (Class 1)"
NEGATIVE_LABEL = "Not diagnosed (Class 0)"

st.markdown(
    f"""
{OUTCOME_FRAMING}

**Dataset:** the bundled Pima Indians Diabetes dataset (`assets/datasets/csv/diabetes.csv`) — 768 records,
8 numeric features, one binary outcome.

**Instructions:** set the test-set size once below, then work through the four activities in order
using the selector. Every model in every activity is trained and scored on that one split. Record your
own responses
wherever your course asks for them.
"""
)


@st.cache_data
def load_diabetes_data():
    return pd.read_csv("assets/datasets/csv/diabetes.csv")


df = load_diabetes_data()
FEATURES = [c for c in df.columns if c != "Outcome"]
X_all = df[FEATURES]
y_all = df["Outcome"]

# One split, shared by every activity, so the slider that sets it lives above the activity picker rather
# than inside activity 1. Streamlit drops a widget's state as soon as the widget stops being rendered:
# leaving this slider inside activity 1 meant the split silently reverted to 0.2 the moment a learner
# moved to activity 2, while the page went on claiming every model used the split they had chosen.
test_size = st.slider(
    "Test set size — held back from training, and shared by all four activities",
    0.1,
    0.5,
    0.2,
    step=0.05,
    help="Share of records held back for testing.",
    key="m4_build_test_size",
)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=test_size, random_state=42, stratify=y_all
)
st.caption(
    f"{len(X_train)} training records, {len(X_test)} testing records. The split is stratified, so the "
    "outcome ratio is preserved on both sides."
)

ACTIVITIES = [
    "1. Data, Split, and Scaling",
    "2. A Model You Can Read",
    "3. A Model With More Capacity",
    "4. Validating and Choosing",
]
# A keyed segmented_control rather than st.tabs: tab selection lives in the browser and is
# lost whenever a widget inside a tab triggers a rerun, which is what sent learners back to
# the first activity mid-edit. This selection is in session_state, so it survives.
activity = st.segmented_control(
    "Activity",
    ACTIVITIES,
    default=ACTIVITIES[0],
    key="m4_build_activity",
    required=True,
)
# ═══════════════════════════════════════════════════════════════════════════
# 1 — Data, split, and scaling
# ═══════════════════════════════════════════════════════════════════════════
if activity == ACTIVITIES[0]:
    st.header("Activity 1: What You Are Actually Feeding the Model")

    st.subheader("Preview")
    n_rows = st.slider(
        "Records to display", 1, 20, 5, help="How many rows of the table to show.", key="m4_build_rows"
    )
    st.dataframe(df.head(n_rows), width="stretch")

    with st.expander("What kind of value is in each column? (data types)"):
        st.write(
            "Before a model can use a column, you have to know what kind of value it holds. The table "
            "below lists every column and the type Python read it as: `int64` and `float64` are whole "
            "and decimal numbers, and `object` would mean text. Eight numeric predictors and one binary "
            "outcome — nothing here is text or a category, which is why this pipeline has no step that "
            "converts labels into numbers."
        )
        # .astype(str): a Series of numpy dtype objects is not Arrow-serializable, and
        # Streamlit would log a conversion traceback before falling back to strings anyway.
        st.dataframe(
            df.dtypes.astype(str).rename("dtype").to_frame(), width="stretch"
        )

    st.subheader("Distributions by outcome")
    feature_to_plot = st.selectbox(
        "Feature to visualize:", FEATURES, help="Look for features whose two curves barely overlap.",
        key="m4_build_feature",
    )
    fig_hist = px.histogram(
        df,
        x=feature_to_plot,
        color="Outcome",
        barmode="overlay",
        title=f"Distribution of {feature_to_plot} by outcome",
        color_discrete_sequence=[cfg.CHART_PRIMARY, cfg.CHART_SECONDARY],  # colourblind-safe
    )
    fig_hist.update_layout(height=380, margin=dict(l=40, r=20, t=55, b=45))
    st.plotly_chart(fig_hist, width="stretch")

    with st.expander("View chart data as text (accessible alternative)"):
        st.dataframe(df.groupby("Outcome")[feature_to_plot].describe())

    st.subheader("The train/test split")
    st.markdown(
        "Every activity uses the split set by the slider at the top of the page. If a model were "
        "evaluated on the data it trained on, it would score its own memorization."
    )
    split_cols = st.columns([1, 1])
    with split_cols[0]:
        split_df = pd.DataFrame(
            {"Set": ["Training", "Testing"], "Count": [len(X_train), len(X_test)]}
        )
        fig_pie = px.pie(
            split_df,
            values="Count",
            names="Set",
            title="Train/test proportions",
            hole=0.4,
            color_discrete_sequence=[cfg.CHART_PRIMARY, cfg.CHART_SECONDARY],
        )
        fig_pie.update_layout(height=330, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig_pie, width="stretch")
        st.caption(
            f"{len(X_train)} training records, {len(X_test)} testing records. The split is stratified, so "
            "the outcome ratio is preserved on both sides."
        )

    with split_cols[1]:
        st.markdown("**Mechanism: what scaling does**")
        st.markdown(
            "Toggle `StandardScaler()` and watch one patient's row. It centres every feature at mean 0 with "
            "unit variance."
        )
        apply_scaler = st.toggle(
            "Apply StandardScaler()",
            value=False,
            help="Simulates sklearn's fit_transform() across the whole feature matrix.",
            key="m4_build_scaler_toggle",
        )

        if apply_scaler:
            scaled = StandardScaler().fit_transform(X_all)
            display_data = scaled[0]
            st.success(
                "Scaled. The numerical gulf between `Glucose` (hundreds) and "
                "`DiabetesPedigreeFunction` (fractions) is gone."
            )
        else:
            display_data = X_all.iloc[0].values
            st.warning(
                "Raw. Fed to a distance- or gradient-based model, `Glucose` dominates purely because its "
                "integers are bigger — not because it matters more."
            )

        st.dataframe(
            pd.DataFrame([display_data], columns=FEATURES, index=["Patient X (row 0)"]).style.format(
                "{:.3f}"
            ),
            width="stretch",
        )
        st.caption(
            "Decision trees are indifferent to this — they split on thresholds, not distances. Tab 3's "
            "network is not."
        )

# ═══════════════════════════════════════════════════════════════════════════
# 2 — Decision tree
# ═══════════════════════════════════════════════════════════════════════════
if activity == ACTIVITIES[1]:
    st.header("Activity 2: A Decision Tree, Start to Finish")
    st.markdown(
        "A decision tree is the one model whose reasoning you can read off the page. Train it, read its "
        "rules, then push a patient through it yourself."
    )

    max_depth = st.slider(
        "Max depth",
        1,
        15,
        4,
        help="How deep the tree may grow. Deeper captures more structure and risks memorizing noise.",
        key="m4_build_depth",
    )

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Test accuracy", f"{accuracy_score(y_test, tree_pred):.4f}")
    metric_cols[1].metric(
        "Training accuracy", f"{accuracy_score(y_train, tree.predict(X_train)):.4f}"
    )
    metric_cols[2].metric("Leaves", f"{tree.get_n_leaves()}")
    st.caption(
        "Push max depth up and watch the two accuracies separate. That gap is overfitting, measured — and "
        "it is the subject of subsection 4.2."
    )

    st.subheader("The tree itself")
    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
    plot_tree(
        tree,
        feature_names=FEATURES,
        class_names=["0", "1"],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax_tree,
    )
    st.pyplot(fig_tree)
    plt.close(fig_tree)

    with st.expander("View the rules as text (accessible alternative)"):
        st.text(export_text(tree, feature_names=FEATURES))

    st.subheader("Live prediction simulator")
    st.markdown(
        "Set a patient's values. The tree routes them through the thresholds above and returns a class and "
        "a probability."
    )
    cfg.try_this("move `Glucose` across the tree's top split and watch the answer flip.")

    input_data = {}
    input_cols = st.columns(4)
    for idx, col in enumerate(FEATURES):
        with input_cols[idx % 4]:
            input_data[col] = st.slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].median()),
                key=f"m4_build_sim_{col}",
            )

    user_df = pd.DataFrame([input_data])[FEATURES]
    prediction = tree.predict(user_df)[0]
    prob = tree.predict_proba(user_df)[0]

    if prediction == 1:
        st.error(f"Model prediction: {POSITIVE_LABEL}")
    else:
        st.success(f"Model prediction: {NEGATIVE_LABEL}")
    st.progress(float(prob[1]), text=f"Predicted probability: {prob[1]*100:.1f}%")

    with st.expander("Reveal expected insights"):
        st.write(
            "At the default max depth of 4, the tree puts `Glucose` and `BMI` near the root. Those primary "
            "splits are the model telling you which measurements carry the most information about the "
            "outcome in this dataset."
        )

# ═══════════════════════════════════════════════════════════════════════════
# 3 — Dense network + mechanism visualisers
# ═══════════════════════════════════════════════════════════════════════════
if activity == ACTIVITIES[2]:
    st.header("Activity 3: When You Need More Capacity")
    st.markdown(
        "A tree can only cut the space with axis-aligned lines. A dense network composes non-linear "
        "combinations of every input at once — which buys accuracy and costs legibility."
    )

    st.subheader("The architecture")
    arch_cols = st.columns([1, 1])
    with arch_cols[0]:
        st.markdown("In Keras this network is written:")
        st.code(
            """
model = Sequential([
    Input(shape=(X_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
            """,
            language="python",
        )
        st.caption(
            "Below we fit the scikit-learn equivalent (`MLPClassifier` with hidden layers 128-64-32), so "
            "the numbers you see are real rather than quoted."
        )

    with arch_cols[1]:
        st.markdown("**Mechanism: what dropout does**")
        dropout_rate = st.slider(
            "Dropout rate",
            0.0,
            0.9,
            0.3,
            step=0.1,
            help="Share of neurons switched off on each training step, so no single pathway becomes "
            "indispensable.",
            key="m4_build_dropout",
        )

        rng = np.random.default_rng(42)
        neurons = np.ones(32)
        neurons[rng.choice(32, int(32 * dropout_rate), replace=False)] = 0
        active_count = int(neurons.sum())

        neuron_html = "<div style='display:flex; gap:10px; flex-wrap:wrap;'>"
        for n in neurons:
            colour = cfg.SUCCESS if n == 1 else cfg.MUTED
            state = "Active" if n == 1 else "Deactivated"
            neuron_html += (
                f"<div style='background-color:{colour}; width:26px; height:26px; border-radius:13px;' "
                f"title='{state} neuron' aria-label='{state} neuron'></div>"
            )
        neuron_html += "</div>"
        st.markdown(neuron_html, unsafe_allow_html=True)
        st.caption(
            f"Of 32 neurons in this layer, {active_count} are active (green) and {32 - active_count} are "
            f"switched off (grey) for this training step."
        )

    st.subheader("Mechanism: the 1-D sliding kernel")
    st.markdown(
        "A `Conv1D` layer does not look at all features at once. It slides a small kernel along the vector, "
        "reading a few adjacent values at a time — which is what makes it suited to signals and sequences "
        "rather than unordered tabular columns."
    )
    st.code(
        """
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    Conv1D(64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
        """,
        language="python",
    )

    kernel_step = st.slider(
        "Slide the kernel (position)",
        1,
        len(FEATURES) - 2,
        1,
        help="Shifts a size-3 kernel along the feature vector, one step at a time.",
        key="m4_build_kernel",
    )

    kernel_values = X_all.iloc[0].values
    kernel_html = "<div style='display:flex; gap:5px; flex-wrap:wrap;'>"
    for i, feat in enumerate(FEATURES):
        in_window = kernel_step - 1 <= i < kernel_step + 2
        if in_window:
            kernel_html += (
                f"<div style='background-color:{cfg.INFO}; color:{cfg.ON_DARK}; padding:10px; border-radius:5px; "
                f"font-weight:bold; text-align:center;' aria-label='Active feature {feat}'>{feat}<br>"
                f"{kernel_values[i]:.2f}</div>"
            )
        else:
            kernel_html += (
                f"<div style='background-color:{cfg.SURFACE_ALT}; color:{cfg.INK}; padding:10px; border-radius:5px; "
                f"text-align:center;' aria-label='Inactive feature {feat}'>{feat}<br>"
                f"{kernel_values[i]:.2f}</div>"
            )
    kernel_html += "</div>"
    st.markdown(kernel_html, unsafe_allow_html=True)
    st.caption(
        "The three blue cells are the values currently multiplied by the kernel's weights to produce one "
        "output. Grey cells are outside the window. Note that these columns have no natural order — which "
        "is precisely why Conv1D is the wrong choice for this particular dataset, and the right one for an "
        "ECG trace."
    )

    st.subheader("Training it, and choosing the operating point")

    train_cols = st.columns([1, 1])
    with train_cols[0]:
        epochs = st.slider(
            "Epochs (max iterations)", 5, 50, 20, help="Passes over the training data.",
            key="m4_build_epochs",
        )
        batch_size = st.select_slider(
            "Batch size", options=[8, 16, 32], value=16,
            help="Records processed before each weight update.", key="m4_build_batch",
        )

    @st.cache_data
    def evaluate_network_folds(max_iter, batch, n_splits=5):
        """5-fold evaluation returning real held-out probabilities per fold."""
        X = X_all.values
        y = y_all.values
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = []
        for train_idx, val_idx in kf.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            net = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=max_iter,
                batch_size=batch,
                random_state=42,
            )
            net.fit(X_tr, y[train_idx])
            folds.append((y[val_idx], net.predict_proba(X_val)[:, 1]))
        return folds

    with st.spinner("Training the network across 5 folds..."):
        fold_results = evaluate_network_folds(epochs, batch_size)

    with train_cols[1]:
        threshold = st.slider(
            "Classification threshold",
            0.1,
            0.9,
            0.5,
            step=0.05,
            help="The probability above which the model calls a case positive.",
            key="m4_build_threshold",
        )

    rows = []
    for y_true, y_prob in fold_results:
        y_pred = (y_prob > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            [
                (tp + tn) / (tp + tn + fp + fn),
                tp / (tp + fn) if (tp + fn) else 0,
                tn / (tn + fp) if (tn + fp) else 0,
                tp / (tp + fp) if (tp + fp) else 0,
            ]
        )
    avg = np.mean(rows, axis=0)

    net_cols = st.columns(4)
    net_cols[0].metric("Avg accuracy", f"{avg[0]:.3f}")
    net_cols[1].metric("Avg sensitivity", f"{avg[1]:.3f}")
    net_cols[2].metric("Avg specificity", f"{avg[2]:.3f}")
    net_cols[3].metric("Avg precision", f"{avg[3]:.3f}")

    st.markdown(
        "These are computed from the network's **actual** held-out probabilities, not from a formula. Move "
        "the threshold and watch sensitivity and specificity move in opposite directions — you are walking "
        "along the ROC curve by hand. There is no threshold that is right in general; there is only the one "
        "that matches what a miss costs versus what a false alarm costs."
    )

    with st.expander("Reveal expected insights"):
        st.write(
            "Accuracy alone is deceptive here: 500 of the 768 records are negative, so a model that always "
            "predicted 'not diagnosed' would score about 65%. Lowering the threshold raises sensitivity and "
            "lowers specificity; the network's advantage over the tree in activity 2 shows up mostly in the "
            "middle of that trade-off, not at the default 0.5."
        )

# ═══════════════════════════════════════════════════════════════════════════
# 4 — Cross-validation and model choice
# ═══════════════════════════════════════════════════════════════════════════
if activity == ACTIVITIES[3]:
    st.header("Activity 4: Cross-Validation, and Which Model to Ship")
    st.markdown(
        "A single split is one draw from a lottery. Cross-validation runs the experiment on every fold, so "
        "you see the spread rather than one number."
    )

    cv_folds = st.slider(
        "Folds", 2, 10, 5, help="How many pieces the dataset is divided into.", key="m4_build_cv_folds"
    )

    scoring = {
        "accuracy": "accuracy",
        "sensitivity": "recall",
        "precision": "precision",
        "specificity": make_scorer(recall_score, pos_label=0),
    }

    with st.spinner("Running cross-validation..."):
        cv_results = cross_validate(
            DecisionTreeClassifier(max_depth=4, random_state=42),
            X_all,
            y_all,
            cv=cv_folds,
            scoring=scoring,
        )

    cv_cols = st.columns(4)
    cv_cols[0].metric("Avg accuracy", f"{np.mean(cv_results['test_accuracy']):.4f}")
    cv_cols[1].metric("Avg sensitivity", f"{np.mean(cv_results['test_sensitivity']):.4f}")
    cv_cols[2].metric("Avg specificity", f"{np.mean(cv_results['test_specificity']):.4f}")
    cv_cols[3].metric("Avg precision", f"{np.mean(cv_results['test_precision']):.4f}")

    cv_df = pd.DataFrame(
        {
            "Fold": [f"Fold {i+1}" for i in range(cv_folds)],
            "Accuracy": cv_results["test_accuracy"],
        }
    )
    fig_cv = px.bar(
        cv_df,
        x="Fold",
        y="Accuracy",
        title="Accuracy per fold (decision tree, depth 4)",
        text_auto=".3f",
        range_y=[0, 1],
        color_discrete_sequence=[cfg.CHART_PRIMARY],
    )
    fig_cv.add_hline(
        y=float(np.mean(cv_results["test_accuracy"])),
        line_dash="dash",
        line_color=cfg.CHART_SECONDARY,
        annotation_text="Mean",
    )
    fig_cv.update_layout(height=380, margin=dict(l=40, r=20, t=55, b=45))
    st.plotly_chart(fig_cv, width="stretch")

    with st.expander("View fold metrics as text (accessible alternative)"):
        st.dataframe(cv_df)

    st.caption(
        f"Spread across folds: {cv_results['test_accuracy'].min():.3f} to "
        f"{cv_results['test_accuracy'].max():.3f}. Any single-split number you report lands somewhere in "
        "that range — which is why one number without a spread is not evidence."
    )

    st.subheader("So which model do you ship?")

    compare_cols = st.columns(2)
    with compare_cols[0]:
        st.markdown("**Decision tree (activity 2)**")
        st.markdown(
            "- Logic: readable if-then thresholds\n"
            "- Transparency: high — you can print the rules\n"
            "- Capacity: axis-aligned splits only"
        )
    with compare_cols[1]:
        st.markdown("**Dense network (activity 3)**")
        st.markdown(
            "- Logic: non-linear combinations across hidden layers\n"
            "- Transparency: low — needs post-hoc explanation\n"
            "- Capacity: high, given enough data"
        )

    priority = st.select_slider(
        "What does your setting actually require?",
        options=["Interpretability", "Balanced", "Performance"],
        key="m4_build_priority",
    )

    if priority == "Interpretability":
        st.info(
            "**Ship the tree.** When a clinician has to justify acting on a prediction, being able to walk "
            "the path from input to output is not a nicety — it is the condition of use."
        )
    elif priority == "Performance":
        st.success(
            "**Ship the network** — and commit to the explainability work in subsection 4.2. Raw predictive "
            "power without SHAP, LIME, and a subgroup audit is a model nobody can defend when it errs."
        )
    else:
        st.warning(
            "**Either, with post-hoc explanation.** The honest balanced answer is a higher-capacity model "
            "plus the accountability tooling from 4.2 — not a mid-capacity model nobody has audited."
        )

st.markdown(
    """
---
**Key takeaways**

- One split, one scaler, one pipeline. Every number above traces back to the split you set at the top of the page.
- Scaling matters for distance- and gradient-based models and not at all for trees. Know which you have.
- The threshold is a decision, not a default. 0.5 encodes the assumption that a miss and a false alarm cost
  the same.
- Capacity is bought with interpretability. Subsection 4.2 is how you buy some of it back.

**Resources:** [scikit-learn decision trees](https://scikit-learn.org/stable/modules/tree.html) ·
[scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/neural_networks_supervised.html) ·
[Keras](https://keras.io/)
"""
)
