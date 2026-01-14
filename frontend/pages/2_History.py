
import streamlit as st
import time
from services.api import APIClient
from utils.state import StateManager
from utils.helpers import show_toast
from components.feature.document_viewer import DocumentViewer
from components.feature.kv_editor import KVEditor
from components.core.status_badge import StatusBadge
from assets.icons import Icons

# --- Page Config ---
st.set_page_config(
    page_title="Editor | Lumina OCR",
    page_icon="⚡",
    layout="wide"
)

def main():
    # 1. Initialize State & Check Context
    StateManager.init_state()
    active_doc_id = StateManager.get_active_document()
    
    if not active_doc_id:
        st.warning("No document selected. Please select a document from the Upload page.")
        if st.button("Go to Upload"):
            st.switch_page("pages/1_Upload.py")
        return

    # 2. Fetch Data
    # Fetch Document Details
    doc_details = APIClient.get_document_details(active_doc_id)
    if not doc_details or doc_details.get("error"):
        st.error("Failed to load document details.")
        return

    # Fetch Extraction Results (if available)
    extraction_data = {}
    current_ext_id = doc_details.get("current_extraction_id")
    
    if current_ext_id:
        extraction_data = APIClient.get_extraction(current_ext_id)
    
    # 3. Sidebar Filtering & Metadata
    with st.sidebar:
        st.markdown(f"### {Icons.EDIT.replace('width=\"16\"', 'width=\"20\"')} Metadata", unsafe_allow_html=True)
        st.markdown(f"**Filename:** {doc_details.get('original_filename')}")
        st.markdown(f"**ID:** `{doc_details.get('id')}`")
        
        status = doc_details.get("status", "pending")
        st.markdown("**Status:**")
        if status == "completed":
            StatusBadge.completed()
        elif status == "processing":
            StatusBadge.processing()
        elif status == "failed":
            StatusBadge.failed(doc_details.get("error_message"))
        else:
            StatusBadge.pending()

        st.markdown("---")
        if st.button("Back to Upload", use_container_width=True):
             st.switch_page("pages/1_Upload.py")

    # 4. Main Split Layout
    col_viewer, col_editor = st.columns([1, 1], gap="medium")
    
    with col_viewer:
        st.subheader("Document Preview")
        # Pass document dict which contains 'file_path'
        DocumentViewer.render(doc_details)
        
    with col_editor:
        st.subheader("Extracted Data")
        
        # Check active processing
        if doc_details.get("status") == "processing":
            st.info("Document is currently processing... Results will appear here automatically.")
            time.sleep(3)
            st.rerun()
            
        elif current_ext_id and extraction_data.get("fields"):
            # Callback for saving edits
            def save_handler(updated_fields):
                success_count = 0
                for field in updated_fields:
                    # We only update if value changed (optimization left for later)
                    # For now, simplistic update loop
                    if field.get("field_value"):
                        res = APIClient.update_field(
                            extraction_data["id"], 
                            field["id"], 
                            field["field_value"]
                        )
                        if res.get("success") is not False:
                            success_count += 1
                
                if success_count > 0:
                    show_toast(f"Updated {success_count} fields successfully!", "success")
                    time.sleep(1)
                    st.rerun()
                else:
                    show_toast("No changes saved or update failed.", "warning")

            KVEditor.render(extraction_data["fields"], on_save=save_handler)
            
        elif doc_details.get("status") == "failed":
             st.error(f"Extraction Failed: {doc_details.get('error_message') or 'Unknown error'}")
             
        else:
             st.info("No extraction results available yet.")

if __name__ == "__main__":
    main()
