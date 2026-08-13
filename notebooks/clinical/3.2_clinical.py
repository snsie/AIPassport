import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import aipassport_config as cfg

# ── Track-specific framing (this file is the clinical track) ────────────────
SCENARIO = "Harmonizing patient data from two hospitals to predict COVID-19 severity"
INSTITUTIONS = ["City_General", "Mountain_View_Clinic"]
ID_PREFIX = "PT"
TARGET_LABEL = "WBC Count"
SECONDARY_LABEL = "O2 Saturation"

st.markdown(
    f"""
Subsection 3.1 decided whether the data was worth using. Now you do the work.

**Scenario:** {SCENARIO}. Two hospitals — **{INSTITUTIONS[0].replace('_', ' ')}** and
**{INSTITUTIONS[1].replace('_', ' ')}** — want to train one model together, and neither is permitted to
send raw patient records to the other.

**The constraint that shapes everything below:** federated learning only works if every site preprocesses
identically. A different outlier rule at one hospital produces model weights that cannot be meaningfully
averaged with the other's. So you will fix your local data first, then share only what is safe to share.
"""
)

with st.expander("Simulation settings", expanded=True):
    c1, c2 = st.columns(2)
    sample_size = c1.slider(
        "Sample size (per institution)",
        50,
        500,
        200,
        step=50,
        help="Simulates larger or smaller local datasets.",
        key="m3_share_n",
    )
    outlier_rate = c2.slider(
        "Outlier contamination",
        0.0,
        0.10,
        0.02,
        step=0.01,
        format="%f",
        help="Share of the dataset carrying erroneous or extreme values.",
        key="m3_share_contamination",
    )


@st.cache_data
def generate_hospital_cohort(n_samples, contamination):
    rng = np.random.default_rng(42)

    n_outliers = int(n_samples * contamination)
    n_regular = n_samples - n_outliers

    # Regular white-cell counts, plus equipment-error outliers
    wbc = np.append(
        rng.normal(7000, 2000, n_regular),
        rng.uniform(50000, 250000, n_outliers),
    )
    rng.shuffle(wbc)  # so the outliers are not all at the end

    return pd.DataFrame(
        {
            "ID": [f"{ID_PREFIX}-{i:03d}" for i in range(n_samples)],
            "Institution": rng.choice(INSTITUTIONS, n_samples),
            "Feature_Target": wbc,
            # Every tenth record is missing its secondary measurement
            "Feature_Secondary": [
                np.nan if i % 10 == 0 else x
                for i, x in enumerate(rng.normal(96, 2, n_samples))
            ],
        }
    )


df_raw = generate_hospital_cohort(sample_size, outlier_rate)

# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Local inspection
# ═══════════════════════════════════════════════════════════════════════════
st.header("Step 1: Local Data Inspection")
st.markdown(
    f"Before collaborating, look at what you actually have. `Feature_Target` is {TARGET_LABEL} — the "
    f"critical model input. `Feature_Secondary` is {SECONDARY_LABEL}, and it has gaps."
)

with st.expander("View the raw dataset", expanded=True):
    st.dataframe(df_raw.head(10), width="stretch")
    miss = int(df_raw["Feature_Secondary"].isna().sum())
    inspect_cols = st.columns(3)
    inspect_cols[0].metric("Records", f"{len(df_raw):,}")
    inspect_cols[1].metric("Missing secondary values", f"{miss}")
    inspect_cols[2].metric(
        f"Max {TARGET_LABEL}", f"{df_raw['Feature_Target'].max():,.0f}"
    )
    st.caption(
        "That maximum is the tell. A physiologically impossible value means an instrument or entry error "
        "somewhere upstream."
    )

# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Outlier detection and handling
# ═══════════════════════════════════════════════════════════════════════════
st.header("Step 2: Outlier Detection and Handling")
st.markdown(
    "Outliers skew federated models badly — a single extreme value shifts one site's weights and drags the "
    "global average with it. Use the z-score to find the statistically improbable points, then decide what "
    "to do with them."
)

col_controls, col_viz = st.columns([1, 2])

with col_controls:
    st.subheader("Configuration")
    z_threshold = st.slider(
        "Z-score threshold",
        1.5,
        5.0,
        3.0,
        step=0.1,
        help="How many standard deviations from the mean before a point is called an outlier. "
        "Lower removes more; higher is more permissive.",
        key="m3_share_zthreshold",
    )

    df_processing = df_raw.copy()
    df_processing["z_score"] = zscore(df_processing["Feature_Target"])
    is_outlier = np.abs(df_processing["z_score"]) > z_threshold

    st.metric(
        "Outliers detected",
        f"{int(is_outlier.sum())}",
        delta=f"{-int(is_outlier.sum())} rows" if is_outlier.any() else "none",
        delta_color="inverse",
    )

    outlier_strategy = st.radio(
        "Handling strategy:",
        ["Remove the rows", "Winsorize (cap at the threshold)"],
        help="Removing is honest but shrinks the cohort. Winsorizing keeps every record at the cost of a "
        "value that was never measured.",
        key="m3_share_strategy",
    )

    apply_filter = st.checkbox(
        "Apply handling and proceed",
        value=False,
        help="Both sites must agree on this rule before either one trains anything.",
        key="m3_share_apply",
    )

with col_viz:
    plot_df = df_processing.reset_index().rename(columns={"index": "Row"})
    plot_df["Status"] = np.where(is_outlier, "Outlier", "Valid data")
    fig = px.scatter(
        plot_df,
        x="Row",
        y="Feature_Target",
        color="Status",
        symbol="Status",
        title=f"Distribution of {TARGET_LABEL}",
        labels={"Feature_Target": TARGET_LABEL},
        color_discrete_map={"Outlier": cfg.DANGER, "Valid data": cfg.CHART_PRIMARY},
    )
    fig.add_hline(
        y=df_processing["Feature_Target"].mean(),
        line_dash="dash",
        line_color=cfg.CHART_TERTIARY,
        annotation_text="Mean",
    )
    fig.update_layout(height=430, margin=dict(l=40, r=20, t=55, b=45))
    st.plotly_chart(fig, width="stretch")

    with st.expander("View chart data as text (accessible alternative)"):
        st.markdown(f"**Summary of {TARGET_LABEL}, split by whether the row was flagged**")
        st.dataframe(
            plot_df.groupby("Status")["Feature_Target"].describe(), width="stretch"
        )
        st.markdown("**The flagged rows themselves**")
        st.dataframe(
            plot_df[plot_df["Status"] == "Outlier"], width="stretch", hide_index=True
        )

if not apply_filter:
    st.warning(
        "Check **Apply handling and proceed** to continue to preprocessing and the federated round."
    )
else:
    if outlier_strategy == "Remove the rows":
        clean_data = df_processing[~is_outlier].copy()
    else:
        lower_limit = df_processing.loc[~is_outlier, "Feature_Target"].min()
        upper_limit = df_processing.loc[~is_outlier, "Feature_Target"].max()
        clean_data = df_processing.copy()
        clean_data["Feature_Target"] = clean_data["Feature_Target"].clip(lower_limit, upper_limit)

    st.success(
        f"{outlier_strategy} applied — {len(clean_data):,} records carried forward "
        f"(from {len(df_raw):,})."
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3 — Imputation and scaling
    # ═══════════════════════════════════════════════════════════════════════
    st.header("Step 3: Imputation and Scaling")
    st.markdown(
        "Both sites must run **identical** preprocessing, or the weights they exchange in Step 4 describe "
        "different quantities and averaging them is meaningless."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Imputation")
        impute_method = st.selectbox(
            "Method for the missing secondary values:",
            ["Mean", "Median", "Zero"],
            help="Mean and median preserve the distribution's centre. Zero introduces a value that means "
            "something clinically — and is almost always wrong.",
            key="m3_share_impute",
        )

        df_imputed = clean_data.copy()
        if impute_method == "Mean":
            fill_val = df_imputed["Feature_Secondary"].mean()
        elif impute_method == "Median":
            fill_val = df_imputed["Feature_Secondary"].median()
        else:
            fill_val = 0

        n_filled = int(df_imputed["Feature_Secondary"].isna().sum())
        df_imputed["Feature_Secondary"] = df_imputed["Feature_Secondary"].fillna(fill_val)
        st.success(f"{n_filled} missing values filled with the {impute_method.lower()} ({fill_val:.2f}).")

    with c2:
        st.subheader("Scaling")
        scaler_type = st.selectbox(
            "Scaler:",
            ["Min-Max Scaler (0 to 1)", "Standard Scaler (z-score)"],
            help="Min-Max squeezes values into [0, 1]. Standard centres them on 0 with unit variance.",
            key="m3_share_scaler",
        )

        scaler = MinMaxScaler() if "Min-Max" in scaler_type else StandardScaler()
        cols_to_scale = ["Feature_Target", "Feature_Secondary"]
        df_final = df_imputed.copy()
        df_final[cols_to_scale] = scaler.fit_transform(df_final[cols_to_scale])
        st.success(f"Both features scaled with the {scaler_type}.")

    st.subheader("Before and after")
    st.markdown(
        f"Left: the raw values, where {TARGET_LABEL} spans four orders of magnitude and "
        f"{SECONDARY_LABEL} is invisible next to it. Right: after handling and scaling, both features "
        "occupy a comparable range — which is what lets a single model weight them fairly."
    )

    box_before = df_raw[cols_to_scale].melt(var_name="Feature", value_name="Value")
    box_before["Feature"] = box_before["Feature"].map(
        {"Feature_Target": TARGET_LABEL, "Feature_Secondary": SECONDARY_LABEL}
    )
    box_after = df_final[cols_to_scale].melt(var_name="Feature", value_name="Value")
    box_after["Feature"] = box_after["Feature"].map(
        {"Feature_Target": TARGET_LABEL, "Feature_Secondary": SECONDARY_LABEL}
    )

    box_cols = st.columns(2)
    with box_cols[0]:
        fig_before = px.box(
            box_before, x="Feature", y="Value", color="Feature", title="Raw (unprocessed)"
        )
        fig_before.update_layout(
            height=380, showlegend=False, margin=dict(l=40, r=20, t=55, b=45)
        )
        st.plotly_chart(fig_before, width="stretch")
    with box_cols[1]:
        fig_after = px.box(
            box_after, x="Feature", y="Value", color="Feature", title=f"Processed ({scaler_type})"
        )
        fig_after.update_layout(
            height=380, showlegend=False, margin=dict(l=40, r=20, t=55, b=45)
        )
        st.plotly_chart(fig_after, width="stretch")

    with st.expander("View chart data as text (accessible alternative)"):
        st.markdown("**Raw (unprocessed)** — five-number summary of each feature")
        st.dataframe(
            box_before.groupby("Feature")["Value"].describe(), width="stretch"
        )
        st.markdown(f"**Processed ({scaler_type})** — the same summary after scaling")
        st.dataframe(box_after.groupby("Feature")["Value"].describe(), width="stretch")

    remaining = int((np.abs(df_final[cols_to_scale].apply(zscore)) > 3).sum().sum())
    st.metric("Values still beyond 3 SD", remaining, delta="clean" if remaining == 0 else "review")

    with st.expander("View the processed data, ready for training"):
        st.dataframe(df_final.head(), width="stretch")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4 — Federated round
    # ═══════════════════════════════════════════════════════════════════════
    st.header("Step 4: A Federated Learning Round")
    st.markdown(
        """
    This simulates the **NVIDIA FLARE** workflow. Instead of sending the cleaned data to a central server:

    1. Each institution trains locally and computes its own weights.
    2. **Only the weights** travel to the aggregator.
    3. The aggregator averages them into a global model and sends it back.

    Nobody ever sees the other site's records.
    """
    )

    if st.button("Run federated learning round", key="m3_share_run_round"):
        inst_names = df_final["Institution"].unique()
        inst_A = df_final[df_final["Institution"] == inst_names[0]]
        inst_B = df_final[df_final["Institution"] == inst_names[1]]

        weights_A = inst_A[cols_to_scale].mean()
        weights_B = inst_B[cols_to_scale].mean()
        global_model = (weights_A + weights_B) / 2

        res1, res2, res3 = st.columns(3)
        with res1:
            st.markdown(f"#### Node A — {inst_names[0].replace('_', ' ')}")
            st.json(weights_A.to_dict())
            st.caption(f"Local weights from {len(inst_A)} private records")
        with res2:
            st.markdown(f"#### Node B — {inst_names[1].replace('_', ' ')}")
            st.json(weights_B.to_dict())
            st.caption(f"Local weights from {len(inst_B)} private records")
        with res3:
            st.markdown("#### Global server")
            st.json(global_model.to_dict())
            st.caption("Aggregated global model")

        st.success("Federated round complete. The global model updated with no data leakage.")
        st.metric(
            "Patient records shared",
            value=0,
            help="Zero rows of raw data left either institution.",
        )
        cfg.try_this(
            "change the z-score threshold or the scaler and run again. The weights move — which is "
            "exactly why both sites have to agree on the preprocessing before the first round, not after."
        )

st.markdown(
    """
---
**Key takeaways**

- Outlier handling is a choice with a cost: removal shrinks the cohort, winsorizing invents a value.
  Either is defensible; silently doing neither is not.
- Imputing with zero is not a neutral default — in clinical data, zero usually means something.
- Scaling is what lets a model weight two features fairly when their units differ by orders of magnitude.
- Federated learning does not remove the need to agree. It moves the agreement earlier: **identical
  preprocessing is the precondition, not a detail.**

**Resources:** [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) ·
[scikit-learn preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
"""
)
