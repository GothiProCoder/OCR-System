
import streamlit as st
from utils.state import StateManager
from utils.helpers import load_css
from assets.icons import Icons

# --- Page Config (Must be the very first Streamlit command) ---
st.set_page_config(
    page_title="Lumina OCR",
    page_icon="⚡", # Browser tab icon can remain emoji/favicon, or use a local image path if strictly needed (keeping emoji for tab for now as it's standard)
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/example/ocr-help',
        'Report a bug': "https://github.com/example/ocr-bugs",
        'About': "# Lumina OCR System\nAdvanced AI-powered Document Extraction."
    }
)

# --- Initialization ---
def main():
    # 1. Initialize State
    StateManager.init_state()

    # 2. Load Global CSS
    load_css("frontend/assets/styles.css")

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
