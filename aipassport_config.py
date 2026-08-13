# AI Passport Central Configuration

import streamlit as st

# Model Settings
DEFAULT_MODEL = "gemma-4-31b-it"
NAVIGATOR_TOOLKIT_BASE_URL = "https://api.ai.it.ufl.edu/v1"

# UI Settings
AI_GUIDE_TITLE = "AIP Guide"
AI_GUIDE_PLACEHOLDER = "Ask the AIP Guide anything..."

# System Instructions
AI_GUIDE_SYSTEM_PROMPT = "You are the AIP Guide, a helpful AI tutor for the AIPassport educational platform. Be concise, encouraging, and accurate."

# ═══════════════════════════════════════════════════════════════════════════
# Brand palette — AI Passport Branding Document v1
# ═══════════════════════════════════════════════════════════════════════════
# This block is the ONLY place a colour is defined. The entrypoint's CSS, the
# AI Guide panel, and every notebook read from here; .streamlit/config.toml
# mirrors four of these values because Streamlit cannot read Python for its
# theme. If you change a value here, change the matching line in config.toml.
#
# Contrast rules the palette imposes (all ratios against the stated pair):
#   - White text is legible on OXFORD_BLUE (14.8:1) and on nothing else here.
#   - AQUAMARINE and HARVEST_GOLD are *backgrounds* for OXFORD_BLUE text
#     (11.7:1 and 7.4:1). White on either fails WCAG AA.
#   - TEAL is the branding document's ADA substitute for AQUAMARINE on web, but
#     only as a fill or rule — white text on TEAL is 3.0:1 and also fails.

OXFORD_BLUE = "#002657"   # main colour
AQUAMARINE = "#70FCE0"    # main accent; print and graphics
TEAL = "#2CA6A4"          # ADA substitute for AQUAMARINE, web only
HARVEST_GOLD = "#F2A900"  # shared third colour on every module palette

# Per-module accents, one page per module in the branding document. Module 1
# has no page of its own and uses the core accent.
MODULE_ACCENTS = {
    1: AQUAMARINE,
    2: "#800020",  # deep red / burgundy — Ethics & Trust
    3: "#002657",  # indigo / navy — Data Integrity & Curation
    4: "#0072CE",  # bright blue — Algorithms & Patterns
    5: "#2E7D32",  # green — Vision & Perception
    6: "#FA4616",  # orange — Creativity & Innovation
    7: "#800080",  # purple — Integration & Real-World Application
}

# ── Semantic roles ─────────────────────────────────────────────────────────
# Notebooks reference these, not the raw brand names, so that a role can be
# retinted without touching 24 files.
INK = "#0E1A2B"          # body text
ON_DARK = "#FFFFFF"      # text on OXFORD_BLUE
SURFACE = "#FFFFFF"
SURFACE_ALT = "#E9F5F3"  # tinted panel; mirrors secondaryBackgroundColor
BORDER = "#D3DEE4"
MUTED = "#5C6B7A"        # de-emphasised series, disabled state

DANGER = MODULE_ACCENTS[2]   # errors, outliers, failed checks
SUCCESS = MODULE_ACCENTS[5]  # active, passing
INFO = MODULE_ACCENTS[4]     # highlighted but neutral

# ── Chart palette ──────────────────────────────────────────────────────────
CHART_PRIMARY = OXFORD_BLUE
CHART_SECONDARY = HARVEST_GOLD
CHART_TERTIARY = TEAL
# Ordered categorical sequence. The lightness spread is wide enough that it
# survives greyscale printing and every common colour-vision deficiency.
CHART_SEQUENCE = [
    OXFORD_BLUE,
    HARVEST_GOLD,
    TEAL,
    MODULE_ACCENTS[7],
    MODULE_ACCENTS[5],
    MUTED,
]


# ═══════════════════════════════════════════════════════════════════════════
# Shared notebook affordances
# ═══════════════════════════════════════════════════════════════════════════
# This module is the only one every notebook already imports, which is why the
# helpers below live here rather than in a package of their own — a notebook is
# exec'd from a string, so it cannot rely on anything the entrypoint happens to
# have in scope.


def try_this(body):
    """Render a 'Try this:' prompt so learners can spot it while scrolling.

    The label carries a gold background and a leading arrow in every notebook,
    so the cue is recognisable before it is read. Both come from Streamlit's
    markdown vocabulary rather than injected CSS: a raw <div> would strip the
    markdown out of *body*, and a font-family rule would turn every Material
    icon on the page into its own literal name.
    """
    st.markdown(f":material/arrow_forward: :orange-background[**Try this:**] {body}")
