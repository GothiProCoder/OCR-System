
import streamlit as st
import time
from services.api import APIClient
from components.feature.upload_zone import UploadZone
from components.core.status_badge import StatusBadge
from utils.state import StateManager
from utils.helpers import format_date
from assets.icons import Icons

# --- Page Config ---
st.set_page_config(
    page_title="Upload | Lumina OCR",
    page_icon="⚡",
    layout="wide"
)

def render_processing_queue():
    """Renders the recent documents list/queue."""
    st.markdown("### Recent Activity")
    
    # Fetch recent docs (limit 5 for the queue view)
    docs_response = APIClient.list_documents(limit=5)
    
    if not docs_response or "items" not in docs_response or not docs_response["items"]:
        st.info("No recent documents found.")
        return

    items = docs_response["items"]
    
    # Auto-refresh if any doc is processing
    if any(d.get("status") == "processing" for d in items):
        time.sleep(2)
        st.rerun()

    for doc in items:
        # Card Container
        with st.container():
            col1, col2, col3, col4 = st.columns([4, 2, 2, 1])
            
            with col1:
                st.markdown(f"**{doc.get('original_filename', 'Unknown File')}**")
                st.caption(f"ID: {doc.get('id')[-8:]} • {format_date(doc.get('created_at'))}")
                
            with col2:
                # Status Badge
                status = doc.get("status", "pending")
                if status == "completed":
                    StatusBadge.completed()
                elif status == "processing":
                    StatusBadge.processing()
                elif status == "failed":
                    StatusBadge.failed()
                else:
                    StatusBadge.pending()
                    
            with col3:
                # Stats (if available)
                if doc.get("form_type"):
                    st.caption(f"Type: {doc.get('form_type')}")
                else:
                    st.caption("Analyzing...")
                    
            with col4:
                # Action Button (View)
                if st.button("Open", key=f"btn_{doc['id']}", help="View Document"):
                    StateManager.set_active_document(doc['id'])
                    st.switch_page("pages/2_History.py")
            
            st.markdown("---")

def main():
    # Initialize State
    StateManager.init_state()
    
    # 1. Header
    st.markdown(
        f"""
        <div style="margin-bottom: 2rem;">
            <h1 style="font-size: 2.25rem; font-weight: 700; color: #0F172A;">Document Upload</h1>
            <p style="color: #64748B;">Upload your documents for intelligent data extraction.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # 2. Layout: Split View
    left_col, right_col = st.columns([5, 4], gap="large")
    
    with left_col:
        # Upload Section
        def on_upload_success():
            # Refresh queue on success
            st.rerun()
            
        UploadZone.render(on_success=on_upload_success)
        
        # Quick Tips
        st.markdown(
            """
            <div style="background-color: #F1F5F9; padding: 1rem; border-radius: 0.5rem; margin-top: 2rem; border: 1px solid #E2E8F0;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #475569;">💡 Quick Tips</h4>
                <ul style="margin: 0; padding-left: 1.25rem; color: #64748B; font-size: 0.85rem;">
                    <li>Ensure images are clear and well-lit.</li>
                    <li>Supported formats: PDF, PNG, JPG, TIFF.</li>
                    <li>Max file size: 20MB.</li>
                </ul>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with right_col:
        # Processing Queue / History
        render_processing_queue()

if __name__ == "__main__":
    main()
