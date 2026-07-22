"""
Theme Manager Module for CurrencyVision AI.
Controls dark and light mode state switching and injects Apple + OpenAI
glassmorphic theme styles into Streamlit.
"""

import os
import streamlit as st


def init_theme_session_state() -> None:
    """Initializes default theme setting in Streamlit session state if not set."""
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"


def toggle_theme() -> None:
    """Toggles current theme between dark and light mode."""
    if st.session_state.get("theme_mode") == "dark":
        st.session_state["theme_mode"] = "light"
    else:
        st.session_state["theme_mode"] = "dark"


def load_custom_css(css_filepath: str = "styles/custom.css") -> None:
    """
    Reads and injects custom CSS file content into Streamlit app.

    Args:
        css_filepath (str): Relative or absolute path to custom.css.
    """
    init_theme_session_state()
    current_mode = st.session_state["theme_mode"]

    if os.path.exists(css_filepath):
        with open(css_filepath, "r", encoding="utf-8") as f:
            css_code = f.read()

        # Inject mode attribute tag wrapper
        theme_wrapper = f"""
        <style>
        :root {{
            --mode: "{current_mode}";
        }}
        {css_code}
        </style>
        """
        st.markdown(theme_wrapper, unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found at '{css_filepath}'. Default styles applied.")


def render_theme_toggle_button() -> None:
    """Renders a sleek theme toggle button in Streamlit sidebar."""
    mode = st.session_state.get("theme_mode", "dark")
    icon = "☀️ Light Mode" if mode == "dark" else "🌙 Dark Mode"

    st.sidebar.markdown("---")
    st.sidebar.caption("🎨 UI Personalization")
    if st.sidebar.button(icon, key="theme_toggle_btn", use_container_width=True):
        toggle_theme()
        st.rerun()
