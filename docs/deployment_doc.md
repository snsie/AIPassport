# AIPassport Deployment & Canvas Integration Guide

This document outlines how to deploy the AI Passport application and embed specific subsections into Canvas.

---

## 🚀 1. Deploying to Streamlit Cloud

1.  **Repository Setup**: 
    - Ensure all changes are committed and pushed to your GitHub repository.
    - The repository structure should have `AIPassport/` as a subfolder containing `aipassport_notebooks.py`.

2.  **Streamlit Cloud Configuration**:
    - Go to [share.streamlit.io](https://share.streamlit.app/) and click **"New app"**.
    - **Repository**: Select your `AIP-Guide` repo.
    - **Branch**: `main`.
    - **Main file path**: `AIPassport/aipassport_notebooks.py`.

3.  **Secrets & Environment**:
    - Click **"Advanced settings..."** before deploying (or go to App Settings > Secrets after deploying).
    - Add your UF NaviGator Toolkit key:
      ```toml
      NAVIGATOR_TOOLKIT_API_KEY = "your_actual_key_here"
      ```
    - This is the only secret the app reads. Without it the app still deploys and every page renders; the
      AI Guide and the LLM-backed activities (1.1's Fact-or-Fiction and all of Module 7) explain that
      feedback is unavailable and disable their inputs.

---

## 📦 2. Embedding in Canvas (Iframe)

Every module presents exactly **two subsections**, at `url_path` `{module}.1` and `{module}.2`. To embed
one with a pre-selected track, use the following `<iframe>` format in the Canvas Rich Text Editor:

### Example: Clinical Track
```html
<iframe 
  src="https://your-app.streamlit.app/1.2?track=clinical&embed=true" 
  width="100%" 
  height="900px" 
  style="border:none;">
</iframe>
```

### Example: Basic Track
```html
<iframe 
  src="https://your-app.streamlit.app/1.2?track=basic&embed=true" 
  width="100%" 
  height="900px" 
  style="border:none;">
</iframe>
```

### URL Parameters Explained:
- **`track=clinical`**: Pre-selects the "Clinical" version of the subsection.
- **`track=basic`**: Pre-selects the "Basic" version.
- **`embed=true`**: Hides the Streamlit header and sidebar for a native "app-like" look.

### The complete set of paths

| Path | Subsection |
| --- | --- |
| `/1.1` | How Does an AI Model Work? |
| `/1.2` | Designing a Study You Can Defend |
| `/2.1` | Ethics, Bias, and Human Oversight |
| `/2.2` | Measuring and Documenting Model Quality |
| `/3.1` | Getting Data You Can Trust |
| `/3.2` | Cleaning and Sharing Data Across Sites |
| `/4.1` | Building a Model End to End |
| `/4.2` | Evaluating and Explaining a Model |
| `/5.1` | How Biomedical Images Become Data |
| `/5.2` | Preprocessing, Features, and Trustworthy Pipelines |
| `/7.1` | From Idea to Study Design |
| `/7.2` | Communicating and Defending Your Work |

Module 6 (Generative AI) has no content and therefore no paths. Both tracks exist for all twelve
subsections, so `?track=` always resolves to a real file.

---

## ⚠️ 2b. Breaking change: old microskill paths were retired

The curriculum was consolidated from up to seven microskills per module down to two subsections per
module. **This changed 25 of the 30 published paths, and there is no redirect layer** — a stale path now
returns Streamlit's "page not found".

**Canvas pages live outside this repository and must be updated by hand.** Use the map below.

| Old path | New path | Where the content went |
| --- | --- | --- |
| `/1.1` | `/1.1` | unchanged path; now also carries the AI lifecycle |
| `/1.2` | `/1.1` | lifecycle simulator folded into 1.1 |
| `/1.3` | `/1.2` | design brief, cut from 22 fields to 6 |
| `/1.4` | `/1.2` | 18 free-text tasks reduced to 4 validation decisions |
| `/1.5` | `/1.2` | reduced to one team prompt plus a shared-vocabulary checklist |
| `/1.6` | `/1.2` | the outlier lab, kept whole |
| `/1.7` | `/1.2` | email-drafting exercise and its worked example |
| `/2.1` | `/2.1` | unchanged path |
| `/2.3` | `/2.1` | Assignment 2 became the bias-vector exercise; Assignment 1 removed |
| `/2.5` | `/2.2` | all three activities, including the Model Card builder |
| `/2.6` | `/2.1` | Question 2 became the human-oversight decision; Question 1 removed |
| `/2.7` | `/2.2` | subgroup performance and both MCQs; odds-ratio stage removed |
| `/3.2` | `/3.1` | consent, representation, and privacy audits |
| `/3.3` | `/3.1` | ICC panel, disagreement table, annotator-count curve |
| `/3.4` | `/3.1` | OMOP concept-ID mapping, now on both tracks |
| `/3.5` | `/3.2` | before/after boxplot and winsorization folded into the outlier step |
| `/3.6` | `/3.2` | inspect, outliers, impute/scale, federated round |
| `/4.1` | `/4.1` | unchanged path |
| `/4.2` | `/4.1` | network training and the real threshold sweep |
| `/4.3` | `/4.1` | was byte-identical to 4.2; deleted |
| `/4.4` | `/4.1` | the three mechanism visualisers; fabricated Phase 3 removed |
| `/4.5` | `/2.2` | duplicated 2.7; deleted, and 2.7's version is the one that survives |
| `/4.6` | `/4.2` | overfitting, tuning, validation strategies |
| `/4.7` | `/4.2` | subgroup fairness, SHAP, LIME, what-if |
| `/5.1` | `/5.1` | unchanged path |
| `/5.2` | `/5.1` and `/5.2` | intensity half to 5.1, augmentation/artifact half to 5.2 |
| `/5.3` | `/5.2` | edges/threshold (basic), texture/morphology (clinical) |
| `/5.4` | `/5.2` | page removed; its application framing became 5.2's closing gate |
| `/5.5` | — | page removed (a bare third-party iframe with no instruction) |
| `/5.6` | `/5.2` | consistency checklist, expanded to six gates |
| `/7.1` | `/7.1` | unchanged path |
| `/7.2` | `/7.2` | NIH-style project summary |
| `/7.3` | `/7.2` | elevator pitch |
| `/7.4` | `/7.1` | gap-to-approach on-ramp |
| `/7.5` | `/7.2` | critique generator |
| `/7.6` | `/7.1` | datasheet / model card generator |
| `/7.7` | `/7.2` | misconduct case, with the case text and Q1/Q2 now shown on the page |

---

## 🏠 3. Navigating the Module Index
If you land on the root URL (`https://your-app.streamlit.app/`), you will see a **Home Page Index**. This dashboard allows you to browse and navigate all available modules manually while the sidebar is hidden.

### To embed the entire directory:
```html
<iframe 
  src="https://your-app.streamlit.app/?embed=true" 
  width="100%" 
  height="1000px" 
  style="border:none;">
</iframe>
```

---

## 🛠️ 4. Maintenance
- **Updating Notebooks**: Edit the `.py` files in `notebooks/clinical/` and `notebooks/basic/`, which follow
  the `{module}.{subsection}_{track}.py` naming convention.
- **Adding or renaming a subsection**: a file is no longer enough on its own. The registration loop walks
  `MODULE_SUBSECTIONS` in `aipassport_notebooks.py` (module name → its two subsection titles), so a new
  subsection needs both the notebook file *and* an entry there. The title in that literal is what the home
  page button and the page header display.
- **Context files**: each notebook has a matching `assets/notebook_context/{module}.{subsection}_{track}.json`.
  The AI Guide reads it verbatim — including `sections[].how_to_use` — so a notebook whose controls change
  needs its context file updated in the same commit, or the tutor will describe widgets that no longer exist.
