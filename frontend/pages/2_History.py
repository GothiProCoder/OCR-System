"""
Lumina OCR - Document Editor
Interactive document review with working navigation.
"""

import streamlit as st
import time
from typing import Dict, Any
from services.api import APIClient
from utils.state import StateManager
from utils.helpers import show_toast
from components.feature.document_viewer import DocumentViewer
from components.feature.working_viewer import render_working_viewer
from components.feature.dropdown_fields import render_dropdown_fields

# --- Page Config ---
st.set_page_config(
    page_title="Editor | Lumina OCR",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === STYLES ===
st.markdown("""
<style>
    /* Clean scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #F1F5F9; border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
    
    /* Sticky viewer */
    [data-testid="stHorizontalBlock"] > div:first-child {
        position: sticky;
        top: 0;
        height: fit-content;
        max-height: 100vh;
        align-self: flex-start;
    }
    
    /* Header */
    .page-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px 0;
        margin-bottom: 16px;
        border-bottom: 2px solid #E2E8F0;
    }
    .header-icon {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .header-text h1 {
        margin: 0;
        font-size: 1.4rem;
        color: #1E293B;
    }
    .header-text p {
        margin: 0;
        color: #64748B;
        font-size: 0.9rem;
    }
    
    /* Status badges */
    .status-completed { color: #16A34A; }
    .status-processing { color: #EAB308; }
    .status-failed { color: #EF4444; }
    .status-pending { color: #64748B; }
</style>
""", unsafe_allow_html=True)


def main():
    # Initialize State
    StateManager.init_state()
    active_doc_id = StateManager.get_active_document()
    
    if 'selected_field' not in st.session_state:
        st.session_state.selected_field = None
    
    # No document selected
    if not active_doc_id:
        st.markdown("""
            <div style="text-align: center; padding: 80px 20px;">
                <div style="
                    background: linear-gradient(135deg, #6366F1, #8B5CF6);
                    width: 64px;
                    height: 64px;
                    border-radius: 16px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 20px auto;
                ">
                    <svg width="32" height="32" fill="white" viewBox="0 0 24 24">
                        <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm4 18H6V4h7v5h5v11z"/>
                    </svg>
                </div>
                <h2 style="color: #1E293B; margin: 0 0 8px 0;">No Document Selected</h2>
                <p style="color: #64748B;">Select a document from Upload to start reviewing</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Upload", type="primary"):
            st.switch_page("pages/1_Upload.py")
        return

    # Fetch Document
    doc_details = APIClient.get_document_details(active_doc_id)
    if not doc_details or doc_details.get("error"):
        st.error("Failed to load document details.")
        return

    # Fetch Extraction
    extraction_data = {}
    current_ext_id = doc_details.get("current_extraction_id")
    
    if current_ext_id:
        extraction_data = APIClient.get_extraction_with_layout(current_ext_id)
    
    # === SIDEBAR ===
    with st.sidebar:
        st.markdown("### 📄 Document Info")
        
        filename = doc_details.get('original_filename', 'Unknown')
        st.markdown(f"**File:** {filename}")
        
        status = doc_details.get("status", "pending")
        status_emoji = {"completed": "✅", "processing": "⏳", "failed": "❌", "pending": "⏸️"}.get(status, "❓")
        st.markdown(f"**Status:** {status_emoji} {status.capitalize()}")
        
        if extraction_data.get("total_fields"):
            st.markdown("---")
            st.metric("Total Fields", extraction_data.get('total_fields', 0))
            conf_avg = extraction_data.get('confidence_avg', 0)
            if conf_avg:
                st.metric("Avg Confidence", f"{int(conf_avg * 100)}%")
            
            # Page info
            total_pages = len(extraction_data.get("processed_image_paths", {})) or 1
            current_page = st.session_state.get("active_viewer_page", 1)
            st.markdown(f"**Page:** {current_page} of {total_pages}")
        
        st.markdown("---")
        if st.button("← Back to Upload", use_container_width=True):
            st.switch_page("pages/1_Upload.py")

    # === HEADER ===
    filename = doc_details.get('original_filename', 'Document')
    st.markdown(f"""
        <div class="page-header">
            <div class="header-icon">
                <svg width="24" height="24" fill="white" viewBox="0 0 24 24">
                    <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
                </svg>
            </div>
            <div class="header-text">
                <h1>Review & Edit Extraction</h1>
                <p>{filename}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # === MAIN LAYOUT ===
    col_viewer, col_editor = st.columns([1.2, 1], gap="large")
    
    with col_viewer:
        has_images = bool(extraction_data.get("processed_image_paths"))
        
        if has_images:
            render_working_viewer(
                doc_id=active_doc_id,
                extraction_data=extraction_data,
                selected_field=st.session_state.selected_field
            )
        else:
            # Fallback to basic viewer
            DocumentViewer.render(doc_details)
    
    with col_editor:
        if doc_details.get("status") == "processing":
            st.info("⏳ Document is processing... Results will appear automatically.")
            time.sleep(3)
            st.rerun()
            
        elif current_ext_id and extraction_data.get("fields"):
            # Save handler
            def save_handler(updated_fields):
                success_count = 0
                for field in updated_fields:
                    if field.get("field_value"):
                        res = APIClient.update_field(
                            extraction_data["id"], 
                            field["id"], 
                            field["field_value"]
                        )
                        if res.get("success") is not False:
                            success_count += 1
                
                if success_count > 0:
                    show_toast(f"Updated {success_count} fields!", "success")
                    time.sleep(1)
                    st.rerun()
            
            # Field select handler
            def field_select_handler(field: Dict[str, Any]):
                st.session_state.selected_field = field

            # Render dropdown fields
            render_dropdown_fields(
                extraction_data["fields"], 
                on_save=save_handler,
                on_field_select=field_select_handler
            )
            
        elif doc_details.get("status") == "failed":
            st.error(f"❌ Extraction Failed: {doc_details.get('error_message') or 'Unknown error'}")
        else:
            st.info("No extraction results available yet.")


if __name__ == "__main__":
    main()
