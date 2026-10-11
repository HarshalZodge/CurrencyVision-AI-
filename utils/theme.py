"""
Theme Manager Module for CurrencyVision AI.
Controls theme state switching and injects light aesthetic styles into Streamlit.
"""

import os
import streamlit as st


def init_theme_session_state() -> None:
    """Initializes default theme setting in Streamlit session state (Default: light)."""
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "light"


def toggle_theme() -> None:
    """Toggles current theme between light and dark mode."""
    if st.session_state.get("theme_mode") == "light":
        st.session_state["theme_mode"] = "dark"
    else:
        st.session_state["theme_mode"] = "light"


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
    mode = st.session_state.get("theme_mode", "light")
    icon = "🌙 Dark Mode" if mode == "light" else "☀️ Light Aesthetic"

    st.sidebar.markdown(
        """
        <div style="margin: 0.8rem 0 0.4rem; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8;">
            Appearance
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button(icon, key="theme_toggle_btn", use_container_width=True):
        toggle_theme()
        st.rerun()
