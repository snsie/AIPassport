#!/usr/bin/env python3
"""Verify the notebook curriculum without a browser.

There is no unit-test suite here, and notebooks are not importable modules -- the
entrypoint reads each one as text and exec()s it. So this script reproduces that
exec path under Streamlit's AppTest and then runs the structural checks that the
architecture makes easy to get wrong.

    python scripts/verify_notebooks.py              # structure + render every notebook
    python scripts/verify_notebooks.py --fast       # structure only, no rendering
    python scripts/verify_notebooks.py --interact   # also click every button and checkbox
    python scripts/verify_notebooks.py 4.1 5.2      # restrict rendering to matching files

Requires the app's own dependencies (pip install -r requirements.txt). An absent
.streamlit/secrets.toml is fine and is itself worth testing: the LLM-backed
activities must degrade gracefully rather than raise.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

ENTRYPOINT = "aipassport_notebooks.py"
# AppTest.from_file() resolves relative paths against the *calling* file, which is
# this script inside scripts/ -- so it needs the absolute path.
ENTRYPOINT_ABS = os.path.join(BASE_DIR, ENTRYPOINT)
NOTEBOOK_GLOB = "notebooks/*/*.py"
CONTEXT_GLOB = "assets/notebook_context/*.json"

CONTEXT_KEYS = {
    "id", "module", "subsection", "microskill", "track",
    "title", "audience", "objectives", "sections", "chatbot_guidance",
}

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)
    print(f"  FAIL  {message}")


def ok(message: str) -> None:
    print(f"  ok    {message}")


def notebooks() -> list[str]:
    return sorted(glob.glob(NOTEBOOK_GLOB), key=lambda p: (os.path.basename(p), p))


def code_files() -> list[str]:
    return notebooks() + [ENTRYPOINT, "aipassport_config.py"] + sorted(
        glob.glob("packages/**/*.py", recursive=True)
    )


# ── The exec harness ────────────────────────────────────────────────────────
# Mirrors render_notebook_page() in the entrypoint: same injected globals, same
# exec into a copy of the module namespace.
HARNESS = '''
import os, sys, json
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = {base!r}
os.chdir(BASE_DIR)
for _p in (os.path.join(BASE_DIR, "packages"), BASE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

def display(*args, **kwargs):
    for arg in args:
        st.write(arg)

class ImageCompatibility:
    def __init__(self, filename=None, data=None, width=None, height=None, **kwargs):
        self.filename, self.data, self.width = filename, data, width
    def _repr_png_(self):
        return self.data

exec_globals = globals().copy()
exec_globals.update({{"display": display, "Image": ImageCompatibility}})

with open(os.environ["AIP_NB"]) as f:
    exec(f.read(), exec_globals)
'''


def app_test(timeout=300):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_string(HARNESS.format(base=BASE_DIR), default_timeout=timeout)


def exceptions_of(at) -> list[str]:
    """Real exceptions only. st.error() is often intentional here -- a failed
    security audit or poor inter-rater agreement is the lesson, not a bug."""
    return sorted({f"{e.type}: {e.message}" for e in at.exception})


# ── 1. Structure: notebooks, context files, and the registration literal ────
def check_structure() -> None:
    print("\n== structure ==")

    nb_stems = {os.path.basename(p)[:-3] for p in notebooks()}
    ctx_stems = {os.path.basename(p)[:-5] for p in glob.glob(CONTEXT_GLOB)}

    if nb_stems - ctx_stems:
        fail(f"notebooks with no context file: {sorted(nb_stems - ctx_stems)}")
    if ctx_stems - nb_stems:
        fail(f"context files with no notebook: {sorted(ctx_stems - nb_stems)}")
    if nb_stems == ctx_stems:
        ok(f"{len(nb_stems)} notebooks and {len(ctx_stems)} context files correspond exactly")

    # MODULE_SUBSECTIONS must describe exactly the files on disk.
    tree = ast.parse(open(ENTRYPOINT).read())
    literal = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MODULE_SUBSECTIONS" for t in node.targets
        ):
            literal = ast.literal_eval(node.value)
    if literal is None:
        fail("MODULE_SUBSECTIONS not found in the entrypoint")
        return

    expected = set()
    for module_idx, (module_name, titles) in enumerate(literal.items(), start=1):
        if titles and len(titles) != 2:
            fail(f"{module_name} declares {len(titles)} subsections; the curriculum promises two")
        for subsection_idx, title in enumerate(titles, start=1):
            if not title.strip():
                fail(f"{module_name} subsection {subsection_idx} has an empty title")
            expected.add(f"{module_idx}.{subsection_idx}")

    on_disk = {stem.split("_")[0] for stem in nb_stems}
    if expected != on_disk:
        fail(f"MODULE_SUBSECTIONS declares {sorted(expected)} but files exist for {sorted(on_disk)}")
    else:
        ok(f"MODULE_SUBSECTIONS matches the {len(expected)} subsections on disk, two per module")

    # Both tracks must exist, so the track selector never silently switches tracks.
    by_subsection = defaultdict(set)
    for stem in nb_stems:
        sub, track = stem.split("_")
        by_subsection[sub].add(track)
    lopsided = {s: sorted(t) for s, t in by_subsection.items() if t != {"basic", "clinical"}}
    if lopsided:
        fail(f"subsections missing a track: {lopsided}")
    else:
        ok("every subsection exists on both tracks")


def check_context_schema() -> None:
    print("\n== context file schema ==")
    problems = 0
    for path in sorted(glob.glob(CONTEXT_GLOB)):
        name = os.path.basename(path)
        stem = name[:-5]
        subsection, track = stem.split("_")
        try:
            doc = json.load(open(path))
        except json.JSONDecodeError as e:
            fail(f"{name}: invalid JSON ({e})")
            problems += 1
            continue

        if missing := CONTEXT_KEYS - set(doc):
            fail(f"{name}: missing keys {sorted(missing)}")
            problems += 1
        if (doc.get("id"), doc.get("subsection"), doc.get("track")) != (stem, subsection, track):
            fail(f"{name}: id/subsection/track disagree with the filename")
            problems += 1
        for section in doc.get("sections", []):
            if set(section) != {"name", "purpose", "how_to_use"}:
                fail(f"{name}: section has keys {sorted(section)}")
                problems += 1
        if not doc.get("sections"):
            fail(f"{name}: no sections, so the AI Guide cannot describe the page")
            problems += 1
    if not problems:
        total = sum(len(json.load(open(p))["sections"]) for p in glob.glob(CONTEXT_GLOB))
        ok(f"all {len(glob.glob(CONTEXT_GLOB))} context files valid ({total} sections described)")


# ── 2. The traps this architecture sets ─────────────────────────────────────
def check_cache_keys() -> None:
    """Notebooks are exec'd from a string, so inspect.getsource() fails for every
    function they define and Streamlit falls back to hashing bytecode -- which
    excludes string constants. Two cached functions sharing a name across the two
    tracks therefore share one cache entry, and whichever ran first wins."""
    print("\n== @st.cache_data / @st.cache_resource keys ==")
    by_name = defaultdict(list)
    for path in notebooks():
        for node in ast.walk(ast.parse(open(path).read())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any("cache_data" in ast.unparse(d) or "cache_resource" in ast.unparse(d)
                   for d in node.decorator_list):
                by_name[node.name].append((path, ast.unparse(node)))

    collisions = 0
    for name, entries in sorted(by_name.items()):
        if len({body for _, body in entries}) > 1:
            fail(f"{name}() has differing bodies in {[p for p, _ in entries]} -- "
                 "rename so each is unique, or they will share one cache entry")
            collisions += 1
    if not collisions:
        ok(f"{sum(len(v) for v in by_name.values())} cached definitions, "
           f"{len(by_name)} names, no colliding bodies")


def check_widget_keys() -> None:
    print("\n== widget keys ==")
    key_re = re.compile(r'key\s*=\s*(?:f)?["\']([^"\']+)["\']')
    dupes = 0
    total = 0
    for path in notebooks():
        keys = key_re.findall(open(path).read())
        total += len(keys)
        repeated = [k for k, n in Counter(keys).items() if n > 1]
        if repeated:
            fail(f"{path}: duplicate keys {repeated} (Streamlit raises DuplicateWidgetID)")
            dupes += 1
    if not dupes:
        ok(f"{total} explicit keys, none duplicated within a file")


def check_page_hygiene() -> None:
    print("\n== page hygiene ==")

    # st.stop() aborts the whole exec, silently hiding everything below it.
    stops = [(p, open(p).read().count("st.stop()")) for p in notebooks() if "st.stop()" in open(p).read()]
    for path, count in stops:
        fail(f"{path}: {count} st.stop() call(s) -- these truncate a merged page; "
             "use an if/else block instead")
    if not stops:
        ok("no st.stop() anywhere in notebooks")

    # set_page_config may only be called once, by the entrypoint.
    configs = [p for p in notebooks() if "set_page_config" in open(p).read()]
    if configs:
        fail(f"set_page_config in {configs} -- only the entrypoint may call it")
    else:
        ok("set_page_config is entrypoint-only")

    # The entrypoint renders the page title; a notebook doing so too duplicates it.
    titled = [p for p in notebooks() if re.search(r'^st\.title\(', open(p).read(), re.M)]
    if titled:
        fail(f"{titled} call st.title() at top level, duplicating the entrypoint's header")
    else:
        ok("no notebook duplicates the entrypoint's page title")


def check_assets() -> None:
    print("\n== asset references ==")
    referenced, missing = set(), []
    literal_re = re.compile(r'["\'](assets/[^"\']+)["\']')
    for path in code_files():
        source = open(path).read()
        for match in literal_re.findall(source):
            referenced.add(match)
            if not os.path.exists(match):
                missing.append((path, match))
        for node in ast.walk(ast.parse(source)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "join"):
                parts = [a.value for a in node.args
                         if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                if parts and parts[0] == "assets" and len(parts) == len(node.args):
                    joined = os.path.join(*parts)
                    referenced.add(joined)
                    if not os.path.exists(joined):
                        missing.append((path, joined))
    for path, target in missing:
        fail(f"{target} referenced by {path} does not exist")
    if not missing:
        ok(f"all {len(referenced)} referenced asset paths resolve on disk")


# ── 3. Rendering ────────────────────────────────────────────────────────────
def check_rendering(filters: list[str], interact: bool) -> None:
    print("\n== rendering ==")
    targets = [p for p in notebooks() if not filters or any(f in p for f in filters)]

    for path in targets:
        os.environ["AIP_NB"] = os.path.join(BASE_DIR, path)
        try:
            at = app_test()
            at.run()
        except Exception as e:  # harness-level explosion
            fail(f"{path}: {type(e).__name__}: {e}")
            continue

        if errs := exceptions_of(at):
            fail(path)
            for message in errs:
                print(f"          {message}")
            continue

        widgets = (len(at.text_area) + len(at.selectbox) + len(at.radio) + len(at.multiselect)
                   + len(at.slider) + len(at.button) + len(at.checkbox) + len(at.text_input))
        ok(f"{path}  ({widgets} widgets)")

        check_activities(path, at)

        if interact:
            check_interactions(path)


def check_activities(path: str, first_run) -> None:
    """Render every option of an activity picker, not just the one it opens on.

    Activities used to be st.tabs, and Streamlit executes every tab body on every
    run -- so one render exercised all of them. They are st.segmented_control now
    and only the selected branch runs, which would leave four fifths of these
    notebooks untested if the harness stopped at the default option."""
    if not first_run.segmented_control:
        return

    for picker in first_run.segmented_control:
        options = list(picker.options)
        for option in options[1:]:  # options[0] is what the plain render already covered
            at = app_test()
            at.session_state[picker.key] = option
            at.run()
            if errs := exceptions_of(at):
                fail(f"{path}: activity {option!r}")
                for message in errs:
                    print(f"          {message}")
            else:
                print(f"        ok  activity {option!r}")


def check_interactions(path: str) -> None:
    """Click each button and tick each checkbox from a fresh run, so a handler
    that raises only when triggered cannot hide behind a clean first render."""
    base = app_test()
    base.run()
    targets = [("button", w.label) for w in base.button]
    targets += [("checkbox", w.label) for w in base.checkbox]

    for kind, label in targets:
        at = app_test()
        at.run()
        widgets = at.button if kind == "button" else at.checkbox
        match = [w for w in widgets if w.label == label]
        if not match:
            continue
        (match[0].click() if kind == "button" else match[0].check()).run()
        if errs := exceptions_of(at):
            fail(f"{path}: {kind} {label!r}")
            for message in errs:
                print(f"          {message}")
        else:
            print(f"        ok  {kind} {label!r}")


def check_entrypoint() -> None:
    """Load the real app and walk from the home page into every subsection, the
    way a learner does. at.switch_page() cannot address function-based st.Pages
    by url_path, so navigation goes through the home-page buttons."""
    print("\n== entrypoint navigation ==")
    from streamlit.testing.v1 import AppTest

    for track in ("clinical", "basic"):
        home = AppTest.from_file(ENTRYPOINT_ABS, default_timeout=300)
        home.query_params["track"] = track
        home.run()
        if errs := exceptions_of(home):
            fail(f"home page ({track}): {errs}")
            continue

        buttons = [b.key for b in home.button if b.key and b.key.startswith("btn_")]
        paths = [k[len("btn_"):] for k in buttons]

        for url_path in paths:
            at = AppTest.from_file(ENTRYPOINT_ABS, default_timeout=300)
            at.query_params["track"] = track
            at.run()
            match = [b for b in at.button if b.key == f"btn_{url_path}"]
            if not match:
                fail(f"/{url_path}: no home-page button")
                continue
            match[0].click().run()
            if errs := exceptions_of(at):
                fail(f"/{url_path}?track={track}")
                for message in errs:
                    print(f"          {message}")
                continue
            titles = [t.value for t in at.title]
            if not any(t.startswith(url_path) for t in titles):
                fail(f"/{url_path}?track={track}: header is {titles}, expected it to lead with the number")
        ok(f"{track}: home page lists {len(paths)} subsections, all of which load")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filters", nargs="*", help="only render notebooks whose path contains these")
    parser.add_argument("--fast", action="store_true", help="skip rendering; structural checks only")
    parser.add_argument("--interact", action="store_true", help="also click buttons and checkboxes")
    args = parser.parse_args()

    check_structure()
    check_context_schema()
    check_cache_keys()
    check_widget_keys()
    check_page_hygiene()
    check_assets()

    if not args.fast:
        check_rendering(args.filters, args.interact)
        if not args.filters:
            check_entrypoint()

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
