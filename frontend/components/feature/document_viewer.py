
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from pathlib import Path
from typing import Dict, Any, Union
import base64
from backend.config import settings

class DocumentViewer:
    """
    Component for viewing documents (PDF/Images) with consistent styling.
    Uses 'streamlit-pdf-viewer' for advanced PDF handling and standard st.image for pictures.
    """

    @staticmethod
    def render(document: Dict[str, Any]):
        """
        Render document preview based on file type.
        
        Args:
            document: Dictionary containing document metadata (must have 'file_path').
        """
        if not document or not document.get("file_path"):
            st.warning("No document selected for preview.")
            return

        # Resolve path (Backend stores relative path strings)
        relative_path = document["file_path"]
        file_path = settings.PROJECT_ROOT / relative_path
        
        if not file_path.exists():
            st.error(f"File not found on server: {relative_path}")
            return

        file_ext = file_path.suffix.lower()

        # Container styling
        st.markdown(
            """
            <div style="
                background-color: white; 
                border-radius: 0.75rem; 
                border: 1px solid #E2E8F0; 
                padding: 1rem; 
                margin-bottom: 1.5rem;
                box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            ">
            """,
            unsafe_allow_html=True
        )

        try:
            if file_ext == ".pdf":
                # Read PDF bytes
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()
                    
                # PDF Viewer with consistent width
                pdf_viewer(input=pdf_bytes, width=700, height=800)
                
            elif file_ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
                # Image Viewer
                st.image(str(file_path), use_container_width=True)
                
            else:
                st.info(f"Preview not available for this file type ({file_ext})")
                
        except Exception as e:
            st.error(f"Error loading preview: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

