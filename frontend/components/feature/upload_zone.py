
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from typing import Optional
from assets.icons import Icons
from services.api import APIClient
from utils.state import StateManager
from utils.helpers import show_toast

class UploadZone:
    """
    Advanced file upload component with Lottie animation and drag-drop interface.
    Handles file selection and triggers API upload.
    """
    
    LOTTIE_UPLOAD_URL = "https://lottie.host/09a0665f-914b-4a58-8951-4091a0aa713c/rZ3q6Z1j5W.json" # Clean cloud upload animation

    @staticmethod
    def _load_lottie_url(url: str):
        try:
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
        except:
            return None

    @staticmethod
    def render(on_success=None):
        """
        Renders the upload zone.
        on_success: Callback function to run after successful upload.
        """
        
        st.markdown(f"""<div class="lumina-card" style="text-align: center; padding: 2rem;">
            <div style="margin-bottom: 1rem; color: #4F46E5;">
                {Icons.NAV_UPLOAD.replace('width="20"', 'width="48"').replace('height="20"', 'height="48"')}
            </div>
            <h3 style="margin-bottom: 0.5rem;">Upload Document</h3>
            <p style="color: #64748B; font-size: 0.9rem; margin-bottom: 1.5rem;">
                Drag and drop your PDF or Image file here.<br>
                We support PDF, PNG, JPG, and TIFF.
            </p>
        </div>""", unsafe_allow_html=True)
        
        # 1. Lottie Animation (Visual Delight)
        lottie_json = UploadZone._load_lottie_url(UploadZone.LOTTIE_UPLOAD_URL)
        if lottie_json:
            st_lottie(lottie_json, height=150, key="upload_anim")

        # 2. File Uploader
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["pdf", "png", "jpg", "jpeg", "tiff"],
            label_visibility="collapsed"
        )

        # 3. Handle Upload Logic
        if uploaded_file is not None:
            # Check if this file is already processed to avoid re-upload loop
            if StateManager.get("last_uploaded_file") != uploaded_file.name:
                
                with st.spinner("🚀 Uploading and starting extraction..."):
                    # Call API
                    result = APIClient.upload_document(uploaded_file, auto_extract=True)
                    
                    if result.get("success", False) or "id" in result:
                        StateManager.set("last_uploaded_file", uploaded_file.name)
                        StateManager.set("active_document_id", result.get("id"))
                        
                        show_toast("Document uploaded successfully!", "success")
                        
                        if on_success:
                            on_success()
                    else:
                        show_toast(f"Upload failed: {result.get('error')}", "error")
