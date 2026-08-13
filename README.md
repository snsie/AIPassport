# AIPassport: AI-Powered Educational Platform

AIPassport is an interactive educational platform designed to teach the fundamentals of Artificial Intelligence, from basic concepts to complex clinical applications. The platform is built with **Streamlit** and integrates the **UF NaviGator Toolkit** to provide real-time tutoring and interactive activities.

---

## 🏗️ Architecture Overview

The project follows a **Monorepo** structure where educational content (notebooks), assets, and internal AI packages are managed in a single repository.

### 1. Global Navigation & Entry Point
- **`aipassport_notebooks.py`**: The main entry point. It handles global navigation using `st.navigation`, UI branding, and the persistent **AIP Guide** sidebar.
- **`aipassport_config.py`**: A central configuration file for managing global constants — `DEFAULT_MODEL`, the NaviGator base URL, system prompts, and UI strings.

### 2. The AIP Guide (AI Tutor)
Instead of a complex external backend, the AI Guide talks straight to the **UF NaviGator Toolkit**:
- **`packages/aip_chat_simple/`**: An internal library that calls the NaviGator endpoint
  (`aipassport_config.NAVIGATOR_TOOLKIT_BASE_URL`) through the **OpenAI-compatible SDK**. Despite the
  `*_gemini_*` filenames in `assets/llm/`, no Google GenAI client is involved.
- **Dynamic Context Sharing**: The guide is "content-aware." It automatically detects the current page from the navigation state and can "see" live activity results (like "Fact or Fiction" verdicts) passed through `st.session_state`.

### 3. Educational Content (Notebooks)
- **`notebooks/clinical/`**: Specialized tracks for medical/clinical AI applications.
- **`notebooks/basic/`**: Core AI/ML concepts for general learners.
- Each notebook is a standalone Streamlit page that can also leverage the central AI configuration.
- **Every module presents exactly two subsections**, `{module}.1` and `{module}.2`, following an
  Understand → Apply arc. Both tracks exist for all twelve subsections. The pairing and the titles live in
  `MODULE_SUBSECTIONS` in `aipassport_notebooks.py`, which is what the registration loop walks — adding a
  subsection requires an entry there as well as a notebook file. See `docs/deployment_doc.md` for the
  full path list.

### 5. Notebook Context (`assets/notebook_context/`)
One JSON per notebook, named `{module}.{subsection}_{track}.json`. The AI Guide receives it verbatim, and
reads `sections[].how_to_use` to tell learners which control to use — so it must be updated whenever a
notebook's controls change.

### 4. Assets & LLM Resources
- **`assets/llm/`**: Contains system instructions and JSON response schemas for structured AI activities.
- **`assets/images/` & `assets/widgets/`**: Static media and JSON data for interactive components like the AI Timeline.

---

## 🛠️ Setup & Local Development

### 1. Prerequisites
- Python 3.12 (see `runtime.txt`)
- A UF NaviGator Toolkit API key

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/bihorac-LAB/AIPassport.git
cd AIPassport

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Secrets Management
The AI Guide and the LLM-backed activities (1.1's Fact-or-Fiction, and all of Module 7) require a
`NAVIGATOR_TOOLKIT_API_KEY`. Create a secrets file:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Then fill in your key:
```toml
# .streamlit/secrets.toml
NAVIGATOR_TOOLKIT_API_KEY = "your-actual-navigator-key-here"
```
*Note: `.streamlit/secrets.toml` is ignored by git to prevent accidental leaks.*

**Without the key the app still runs.** Every LLM-backed activity detects the missing key, explains why
feedback is unavailable, and disables its input rather than raising — so the rest of the curriculum is
fully usable for local development.

---

## 🚀 Running the App
```bash
streamlit run aipassport_notebooks.py
```

---

## ✅ Verifying Changes
There is no unit-test suite: notebooks are not importable modules, because the entrypoint reads each one as
text and `exec()`s it. `scripts/verify_notebooks.py` reproduces that exec path under Streamlit's `AppTest`
and adds the structural checks this architecture makes easy to get wrong.

```bash
python scripts/verify_notebooks.py            # structure + render all 24 notebooks (~1 min)
python scripts/verify_notebooks.py --fast     # structural checks only (~2 s)
python scripts/verify_notebooks.py --interact # also click every button and checkbox (~4 min)
python scripts/verify_notebooks.py 4.1 5.2    # restrict rendering to matching files
```

It checks that notebooks and context files correspond one-to-one, that `MODULE_SUBSECTIONS` matches the
files on disk with both tracks present, that no two cached functions collide (see below), that no widget key
repeats within a file, that no notebook calls `st.stop()` or `set_page_config` or duplicates the page title,
that every `assets/` reference resolves, and that every subsection loads through the real entrypoint on both
tracks.

**One trap worth knowing about.** Because notebooks are `exec`'d from a string, `inspect.getsource()` fails
for every function they define, so Streamlit's cache falls back to hashing **bytecode** — which excludes
string constants. Two `@st.cache_data` functions that share a name across the two tracks will therefore
share one cache entry, and whichever ran first wins. Give every cached function a name unique to its body;
the verify script enforces this.

---

## 🎨 Branding
The platform follows the **AI Passport Branding Document v1**. Every colour is defined once, in the brand
palette block of `aipassport_config.py`; notebooks and components import it as `cfg` and reference the
semantic roles (`cfg.CHART_PRIMARY`, `cfg.DANGER`, …) rather than hex literals.

- **Oxford Blue** `#002657` — main colour, and the Streamlit `primaryColor`
- **Aquamarine** `#70FCE0` — main accent (print/graphics); on the web its ADA-compliant substitute
  **Teal** `#2CA6A4` is used instead
- **Harvest Gold** `#F2A900` — the shared third colour on every module palette
- `cfg.MODULE_ACCENTS` holds the per-module accent from the branding document (e.g. Module 2 burgundy,
  Module 7 purple)
- Type is **IBM Plex Sans**, the branding document's face for UF websites and Canvas, loaded by CSS in
  the entrypoint

`.streamlit/config.toml` duplicates four of these values because Streamlit's theme cannot read Python —
change `aipassport_config.py` first, then mirror it there.
