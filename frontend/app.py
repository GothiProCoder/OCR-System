"""
Frontend Application Entry Point
================================
Main Streamlit application with page routing and global configuration.
"""

import streamlit as st
from pathlib import Path
from utils.state import StateManager
from utils.helpers import load_css
from assets.icons import Icons

# --- Page Config (Must be the very first Streamlit command) ---
st.set_page_config(
    page_title="Lumina OCR",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/example/ocr-help',
        'Report a bug': "https://github.com/example/ocr-bugs",
        'About': "# Lumina OCR System\nAdvanced AI-powered Document Extraction."
    }
)

# Get the directory of this file for relative path resolution
FRONTEND_DIR = Path(__file__).parent


# --- Initialization ---
def main() -> None:
    """Initialize and run the main application."""
    # 1. Initialize State
    StateManager.init_state()

    # 2. Load Global CSS (using absolute path for reliability)
    css_path = FRONTEND_DIR / "assets" / "styles.css"
    load_css(str(css_path))


    # 3. Sidebar Header
    with st.sidebar:
        # Render Logo SVG
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px;">
                {Icons.logo(size=32)}
                <h1 style="margin: 0; font-size: 24px; color: #0F172A;">Lumina OCR</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            '<div style="margin-bottom: 1.5rem; color: #64748B; font-size: 0.9rem; padding-left: 4px;">'
            'Intelligent Document Processing'
            '</div>', 
            unsafe_allow_html=True
        )

    # 4. Routing Logic
    st.switch_page("pages/1_Upload.py")

if __name__ == "__main__":
    main()
