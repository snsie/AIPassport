import streamlit as st
from openai import OpenAI
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import base64

import aipassport_config as cfg

def is_rate_limit(exception):
    return "429" in str(exception)

@retry(
    retry=retry_if_exception(is_rate_limit),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def generate_with_retry(client, model_id, messages):
    return client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.7
    )

def render_ai_guide(navigator_api_key: str, context_fn=None):
    """
    Renders a NaviGator-powered chat interface.
    Render order:
      1. Quick-action buttons (pinned at top)
      2. Message history (scrollable)
      3. st.chat_input (stBottom — always stays at the true bottom)
    """
    if not navigator_api_key:
        st.error("Missing NAVIGATOR_TOOLKIT_API_KEY in secrets.")
        return

    st.markdown(
        f"""
        <div style="padding: 0.25rem 0 0.75rem 0; border-bottom: 2px solid {cfg.TEAL}; margin-bottom: 0.75rem;">
            <div style="font-size: 1.15rem; font-weight: 700; color: {cfg.OXFORD_BLUE};">{cfg.AI_GUIDE_TITLE}</div>
            <div style="font-size: 0.85rem; color: {cfg.MUTED}; margin-top: 0.15rem;">Notebook help and activity guidance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    client = OpenAI(
        api_key=navigator_api_key,
        base_url=cfg.NAVIGATOR_TOOLKIT_BASE_URL
    )
    model_id = cfg.DEFAULT_MODEL

    # ── Session state ────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_error" not in st.session_state:
        st.session_state.chat_error = None

    # ── 1. Error banner ───────────────────────────────────────────────────────
    if st.session_state.chat_error:
        st.error(st.session_state.chat_error)
        if st.button("Dismiss Error"):
            st.session_state.chat_error = None
            st.rerun()

    # ── 2. Message history ────────────────────────────────────────────────────
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("Hello! I am your AIP Guide. How can I help you today?")

    for message in st.session_state.messages:
        role = message["role"]
        if role == "system":
            continue
        with st.chat_message(role):
            content = message["content"]
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        st.markdown(item["text"])
                    elif item["type"] == "image_url":
                        st.image(item["image_url"]["url"])
            else:
                st.markdown(content)

    # ── 3. Quick-action buttons (just above chat input) ───────────────────────
    st.markdown("**Quick help**")
    if st.button("How do I use this activity?", width="stretch"):
        st.session_state["_quick_action"] = "How do I use this activity and what controls are available?"
        st.rerun()
    # Only 1.1 publishes _live_state, so on every other page the Guide cannot actually see the learner's
    # values. Asking it to "explain what's on my screen" there invited it to invent numbers, which is the
    # worst failure mode a tutor has. The wording below asks what it can answer from the page's controls.
    if st.button("What am I looking at?", width="stretch"):
        st.session_state["_quick_action"] = (
            "Explain what this part of the notebook is showing and what the controls do. If you have not "
            "been given my current on-screen values, say so and ask me what I am seeing rather than "
            "guessing at specific numbers."
        )
        st.rerun()
    if st.button("Let's do this step by step", width="stretch"):
        st.session_state["_quick_action"] = "Walk me through this notebook step by step. Start with what I should do first, then wait for my next question."
        st.rerun()

    # ── 4. Chat input (renders in stBottom — always at the true bottom) ───────
    prompt = st.chat_input(cfg.AI_GUIDE_PLACEHOLDER)

    # Consume quick action if set
    if "_quick_action" in st.session_state:
        prompt = st.session_state.pop("_quick_action")

    # ── 5. Handle prompt ──────────────────────────────────────────────────────
    if prompt:
        # Temporarily display the user message while the API call is in progress
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build messages list for API
        api_messages = [{"role": "system", "content": cfg.AI_GUIDE_SYSTEM_PROMPT}]

        context_str = ""
        if context_fn:
            try:
                ctx = context_fn()
                context_str += f"\n[Functional Context: {ctx}]"
            except Exception as e:
                context_str += f"\n[Context Error: {e}]"

        # Say explicitly when there is no live state. Left unsaid, the model treats the static section
        # descriptions as though they were a readout of the learner's screen and answers with numbers it
        # never received.
        if live_state := st.session_state.get("_live_state"):
            context_str += f"\n[Live Screen State: {live_state}]"
        else:
            context_str += (
                "\n[Live Screen State: unavailable. You cannot see this learner's current values, chart"
                " output, or selections. Describe what the page offers and ask them what they see;"
                " never state a specific on-screen number as though you had read it.]"
            )

        user_content = [{"type": "text", "text": prompt + context_str}]

        if img_data := st.session_state.get("_screen_image"):
            try:
                b64_img = base64.b64encode(img_data).decode("utf-8") if isinstance(img_data, bytes) else img_data
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}})
            except Exception as e:
                user_content[0]["text"] += f"\n[Image Error: {e}]"

        # Include last 10 historical messages to avoid token blowup
        api_messages.extend(st.session_state.messages[-10:])
        api_messages.append({"role": "user", "content": user_content})

        # Generate response inline
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                response = generate_with_retry(client, model_id, api_messages)
                full_response = response.choices[0].message.content

                # Simulate streaming
                curr = ""
                for w in full_response.split():
                    curr += w + " "
                    placeholder.markdown(curr + "▌")
                    time.sleep(0.01)
                placeholder.markdown(full_response)

                # Save to history and rerun so the history loop re-renders cleanly
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.chat_error = None
                st.rerun()

            except Exception as e:
                st.session_state.chat_error = f"Sorry, I encountered an error: {e}"
                st.rerun()
