# CLAUDE.md

Streamlit teaching platform (UF AI Passport). 7 modules × 2 subsections × 2 tracks
(`clinical`/`basic`) = 24 notebooks, each `exec()`'d as text by one entrypoint.

## Commands

```bash
source venv/bin/activate                      # Python 3.12; venv/ is gitignored
streamlit run aipassport_notebooks.py         # run the app

python scripts/verify_notebooks.py            # full check: structure + render all 24 (~1 min)
python scripts/verify_notebooks.py --fast     # structural checks only (~2 s) — use while iterating
python scripts/verify_notebooks.py --interact # also clicks every button/checkbox (~4 min)
python scripts/verify_notebooks.py 4.1 5.2    # restrict *rendering* to matching paths
```

- No pytest suite, no linter, no formatter, no CI for code. The only GitHub Action
  (`keep_alive.yml`) curls the deployed app every 6h to stop Streamlit Cloud hibernation.
- `verify_notebooks.py` **is** the test suite. Run `--fast` after any structural edit, full before commit.
- Passing filters skips `check_entrypoint()` — a full run with no args is the only thing that
  exercises real navigation.
- A render also walks every option of each activity picker (`check_activities`). Only the selected
  branch of a `st.segmented_control` executes, so without that pass four fifths of 2.2/3.1/4.x/5.x
  would never run. Don't remove it.

## Architecture

- `aipassport_notebooks.py` — the whole app. Registers pages, renders header + track selector, then
  reads the notebook file as **text** and `exec()`s it into a copy of its own globals.
- Notebooks are *not* importable modules. They are top-level scripts with no `main()`, nothing called
  on import, no `if __name__`. They inherit the entrypoint's globals plus injected `display()` and
  `Image` (Jupyter compatibility shims).
- Routing is filename convention, not a registry:
  `url_path "4.1"` + `track "clinical"` → `notebooks/clinical/4.1_clinical.py`
  → `assets/notebook_context/4.1_clinical.json`.
- Adding a subsection takes **three** edits: the notebook file (both tracks), the context JSON (both
  tracks), and a title in `MODULE_SUBSECTIONS` in the entrypoint. `verify_notebooks.py` parses that
  literal with `ast` and fails if the three disagree.
- Module 6 is intentionally empty (`[]` in `MODULE_SUBSECTIONS`); the home page skips pageless modules.
- `packages/aip_chat_simple/` — the sidebar AI Guide. `packages/` is prepended to `sys.path` at startup.
- All LLM traffic goes to the **UF NaviGator** endpoint via the **OpenAI SDK** (`aipassport_config.py`:
  `NAVIGATOR_TOOLKIT_BASE_URL`, `DEFAULT_MODEL`). One secret: `NAVIGATOR_TOOLKIT_API_KEY`.
- The Guide is content-aware: it reads the current page from `pg`, loads the matching context JSON,
  and picks up `st.session_state["_live_state"]` (currently written only by 1.1) as screen state.

## Conventions

- **Colour lives in exactly one place: the brand palette block in `aipassport_config.py`.** Notebooks
  do `import aipassport_config as cfg` and use semantic roles — `cfg.CHART_PRIMARY`,
  `cfg.CHART_SECONDARY`, `cfg.DANGER`, `cfg.SUCCESS`, `cfg.MUTED`, `cfg.SURFACE_ALT`, `cfg.BORDER`,
  `cfg.INK`. No hex literal belongs in a notebook; `grep -rP '#[0-9a-f]{6}' notebooks/` should stay empty.
- Palette is the **AI Passport Branding Document v1**, not UF's: Oxford Blue `#002657` (main),
  Aquamarine `#70FCE0` (accent), Teal `#2CA6A4` (accent's ADA substitute on web), Harvest Gold
  `#F2A900`. `cfg.MODULE_ACCENTS` holds the per-module accent. Type is IBM Plex Sans.
- **All typography and theming lives in `.streamlit/config.toml`, never in injected CSS.** 1.61 has a
  full theme API (`font`, `headingFont`, `codeFont`, `borderColor`, `linkColor`,
  `chartCategoricalColors`, `baseRadius`, …). `font` takes a `"Name:https://…css2?family=…"` string
  and loads the webfont itself.
- `theme.chartCategoricalColors` is set to `cfg.CHART_SEQUENCE`, so a Plotly/Altair/Vega chart with
  **no** explicit colours is already on-brand. Only pass `color_discrete_map`/`_sequence` when a
  series carries meaning (outlier → `cfg.DANGER`, passing → `cfg.SUCCESS`).
- Widget keys: `m{module}_{activity}_{field}` — e.g. `m1_fof_input`, `m7_design_submit`. Keys are
  **deliberately identical across tracks** (only one track renders per request), so copy them verbatim
  when you edit a track's twin.
- Section banners inside notebooks use the `═══` box-comment style with `# Part N — Title` or
  `# N — Title`. Headings use Material icons: `st.markdown("## :material/psychology: ...")`.
- Activities are wrapped in `with st.container(border=True):`.
- **Multi-activity notebooks use a keyed `st.segmented_control`, never `st.tabs`.** The list of labels
  goes in an `ACTIVITIES`/`AUDITS`/`OPERATIONS` constant, the picker carries `key=` and `required=True`
  (so the active chip cannot be deselected into an empty page), and each body is
  `if activity == ACTIVITIES[n]:`. Tab selection is browser-side and is lost on any rerun triggered from
  inside a tab, which used to throw learners back to activity 1 mid-edit.
- Two consequences of that, both of which have already caused bugs:
  1. **Only the selected branch executes**, so anything two activities share — a train/test split, a
     fitted model — must be computed above the picker. See 4.2's `fair_model`.
  2. **A widget inside a branch loses its state the moment the learner leaves that branch.** Streamlit
     drops `session_state` for any widget that stops rendering, so reading a widget key from above the
     picker silently reverts to the default one rerun later. A control whose value the whole page
     depends on belongs *above* the picker, as a widget — that is why 4.1's test-size slider and 4.2's
     dataset selectbox live there.
- Prompts that ask the learner to go move something use `cfg.try_this("…")`, which renders one
  consistent arrow-plus-gold-label cue. Don't hand-roll the styling; the helper is in
  `aipassport_config.py` and deliberately uses Streamlit's markdown vocabulary, not CSS.
- **Every data chart carries a text alternative** in an expander labelled exactly
  `"View chart data as text (accessible alternative)"`, holding the same numbers as a table. Image
  displays are exempt; histograms of images are not.
- Learner-facing prompt headings are `Consider this:` with a `Your response:` input — never
  "Reflection"/"Your reflection", which reads as the separate course reflection journal.
- Every notebook ends with a `**Key takeaways**` bullet list. 1.2 is the model to copy.
- Track-specific wording is hoisted to ALL_CAPS constants at the top of the file
  (`OUTCOME_FRAMING`, `POSITIVE_LABEL`) so the two track files stay diffable.
- Context JSON schema is enforced: exactly `{name, purpose, how_to_use}` per section, and
  `id`/`subsection`/`track` must match the filename. `how_to_use` names the actual controls — update it
  whenever you change a notebook's widgets.
- `assets/llm/*_gemini_*` filenames are historical. No Google GenAI client exists in this codebase.
- Docstrings carry *why*, not *what* — see `run_llm_activity` in the 7.x notebooks.

## Gotchas

- **Cached function names must be globally unique across all 24 notebooks unless the bodies are
  byte-identical.** Because notebooks are `exec`'d from a string, `inspect.getsource()` fails and
  Streamlit hashes bytecode instead — which excludes string constants. Two same-named
  `@st.cache_data` functions with different bodies silently share one cache entry, first-run-wins.
  Hence `load_intensity_samples_basic` / `_clinical` carry track suffixes while `load_diabetes_data`
  (identical in both) does not. Enforced by `check_cache_keys()`.
- **Never call `st.stop()` in a notebook** — it aborts the entrypoint's `exec` and silently blanks
  everything below it, including the chat panel. Use `if/else`.
- **Never call `st.set_page_config()` or top-level `st.title()` in a notebook** — the entrypoint owns
  both; a notebook title duplicates the header.
- The LLM submit path uses a **pending-flag + `st.rerun()`** pattern (button sets flag → rerun →
  generate). It looks like a redundant extra rerun; it is not. Collapsing it breaks button-triggered
  streaming. See the docstring in `notebooks/*/7.1_*.py`.
- Notebooks must degrade without a key: `client = OpenAI(...) if navigator_api_key else None`,
  `disabled=client is None` on inputs, `st.warning` explaining why. Verification runs with **no**
  `secrets.toml` on purpose, so hard-failing on a missing key breaks the checks.
- `st.error()` in a notebook is often intentional pedagogy (a failed audit, poor inter-rater
  agreement). `verify_notebooks.py` only fails on real exceptions, not `st.error`.
- Asset paths are **relative** (`"assets/datasets/csv/diabetes.csv"`), resolved against CWD = repo
  root. The verify harness `os.chdir`s to make this hold. Don't switch to `__file__`-relative paths —
  `__file__` is not meaningful inside an `exec`'d string.
- `.streamlit/config.toml` duplicates palette values because Streamlit's theme cannot read Python.
  Edit `aipassport_config.py` first, then mirror. `primaryColor` is Oxford Blue, *not* the brand
  accent, because Streamlit puts white text on primary buttons and neither Aquamarine nor Teal
  carries white text at AA. Aquamarine and Harvest Gold are backgrounds for Oxford Blue text only.
- **Never set `font-family` in injected CSS.** Streamlit draws Material icons as *ligatures* on spans
  that carry `st-emotion-cache-*` classes, so a broad rule like
  `[class*="st-"] { font-family: … !important }` makes every icon render as its literal name —
  `keyboard_arrow_down` sprawling across each expander header, and every `:material/…:` heading icon
  in the notebooks. Symptom: icon text where a glyph belongs. Use `theme.font` instead.
- The entrypoint's first `st.markdown` CSS block is an **f-string** (all literal braces doubled) so it
  can interpolate `cfg` colours into `:root` custom properties. Every later block is a plain string
  referring to `var(--ap-*)`. Don't add a raw colour to those later blocks.
- To hide or target a Streamlit element, give the container a key —
  `st.container(key="aip_toggle_host")` emits class `st-key-aip_toggle_host`. Wrapping a widget in a
  raw `<div id=…>` from `st.markdown` **does not work**: each markdown call renders in its own
  container, the browser auto-closes the div, and the widget lands outside it. That bug left a live
  "Toggle Chat" button visible at the top of every page. The chat toggle's CSS, its JS
  `querySelector`, and the container key must stay in sync.
- `requirements.txt` pins `streamlit>=1.31.0`, but the code needs **≥1.36** (`st.navigation`) and
  really 1.6x (`st.segmented_control`, `st.spinner(show_time=)`). The floor is knowingly stale; see
  the comment in the file.
- The entrypoint references `reference/demos/aip_streamlit_demo.py`, which is not in the repo.
  Guarded by `os.path.exists`, so it's dead code, not a bug.
- `docs/aip_guide_architecture.md` is a 0-byte placeholder.
