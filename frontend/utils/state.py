
import streamlit as st
from typing import Dict, Any, Optional

class StateManager:
    """
    Manages Streamlit session state with type safety and default initialization.
    Acts as a central store for frontend state.
    """
    
    DEFAULTS = {
        "active_document_id": None,      # Currently selected doc UUID
        "processing_queue": [],          # List of docs currently processing
        "upload_status": "idle",         # idle, uploading, success, error
        "user_preferences": {
            "auto_process": True,
            "theme": "light"
        }
    }

    @staticmethod
    def init_state():
        """Initialize session state with defaults if missing."""
        for key, default_value in StateManager.DEFAULTS.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    @staticmethod
    def get(key: str) -> Any:
        """Get a value from session state safely."""
        return st.session_state.get(key, StateManager.DEFAULTS.get(key))

    @staticmethod
    def set(key: str, value: Any):
        """Set a value in session state."""
        st.session_state[key] = value

    @staticmethod
    def set_active_document(doc_id: str):
        """Set the currently active document for editing/viewing."""
        st.session_state["active_document_id"] = doc_id
    
    @staticmethod
    def get_active_document() -> Optional[str]:
        return st.session_state.get("active_document_id")
