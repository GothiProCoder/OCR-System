
import requests
import streamlit as st
from typing import Dict, Any, Optional, List
from backend.config import settings # Import settings directly or hardcode if running separately

# Base URL for Backend API
API_BASE_URL = "http://localhost:8000/api"

class APIClient:
    """
    Client for interacting with the OCR Backend API.
    Handles error catching and session timeouts.
    """
    
    @staticmethod
    def _handle_response(response: requests.Response) -> Dict[str, Any]:
        """Process API response and handle errors."""
        try:
            if response.status_code in [200, 201]:
                return response.json()
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                return {"error": response.text, "success": False}
        except Exception as e:
            st.error(f"Failed to parse response: {str(e)}")
            return {"error": str(e), "success": False}

    # --- Document Endpoints ---
    
    @staticmethod
    def upload_document(file, auto_extract: bool = True) -> Dict[str, Any]:
        """Upload a file to the backend."""
        try:
            files = {"file": (file.name, file, file.type)}
            data = {
                "auto_extract": str(auto_extract).lower()  # Convert bool to string for form-data
            }
            response = requests.post(f"{API_BASE_URL}/documents/upload", files=files, data=data)
            return APIClient._handle_response(response)
        except Exception as e:
            return {"error": f"Connection failed: {str(e)}", "success": False}

    @staticmethod
    def list_documents(limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """Get list of uploaded documents."""
        try:
            response = requests.get(f"{API_BASE_URL}/documents", params={"limit": limit, "skip": skip})
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []

    # --- Extraction Endpoints ---
    
    @staticmethod
    def get_extraction_status(doc_id: str) -> Dict[str, Any]:
        """Get extraction status/result for a document."""
        # Note: Backend doesn't have a direct /status endpoint on doc ID, 
        # usually we fetch the doc or the extraction. Assuming we fetch latest extraction.
        # This might need adjustment based on backend specific routes.
        # For now, fetching document details which includes status.
        try:
            response = requests.get(f"{API_BASE_URL}/documents/{doc_id}")
            return APIClient._handle_response(response)
        except:
            return {"status": "error"}

    @staticmethod
    def get_document_details(doc_id: str) -> Dict[str, Any]:
        """Get details of a specific document."""
        try:
            response = requests.get(f"{API_BASE_URL}/documents/{doc_id}")
            return APIClient._handle_response(response)
        except Exception as e:
            return {"error": str(e), "success": False}

    @staticmethod
    def get_extraction(extraction_id: str) -> Dict[str, Any]:
        """Get full extraction results including fields."""
        try:
            response = requests.get(f"{API_BASE_URL}/extractions/{extraction_id}")
            return APIClient._handle_response(response)
        except Exception as e:
            return {"error": str(e), "success": False}

    @staticmethod
    def update_field(extraction_id: str, field_id: str, value: str) -> Dict[str, Any]:
        """Update a specific field's value."""
        try:
            data = {"field_value": value}
            response = requests.patch(
                f"{API_BASE_URL}/extractions/{extraction_id}/fields/{field_id}",
                json=data
            )
            return APIClient._handle_response(response)
        except Exception as e:
            return {"error": str(e), "success": False}

    @staticmethod
    def get_dashboard_stats(period: str = "week") -> Dict[str, Any]:
        """Get comprehensive dashboard statistics."""
        try:
            response = requests.get(f"{API_BASE_URL}/stats/dashboard", params={"period": period})
            return APIClient._handle_response(response)
        except Exception as e:
             return {"error": str(e), "success": False}

    @staticmethod
    def get_dashboard_stats(period: str = "week") -> Dict[str, Any]:
        """Get comprehensive dashboard statistics."""
        try:
            response = requests.get(f"{API_BASE_URL}/stats/dashboard", params={"period": period})
            return APIClient._handle_response(response)
        except Exception as e:
            return {"error": str(e), "success": False}
    
    # --- Bounding Box Overlay Endpoints ---
    
    @staticmethod
    def get_processed_image_url(doc_id: str, page_number: int = 1) -> str:
        """
        Get URL for processed image (for bounding box overlay).
        Returns the direct URL to fetch the image.
        """
        return f"{API_BASE_URL}/documents/{doc_id}/processed-image/{page_number}"
    
    @staticmethod
    def get_processed_image_bytes(doc_id: str, page_number: int = 1) -> Optional[bytes]:
        """
        Fetch processed image bytes directly.
        Used when displaying image with Streamlit.
        """
        try:
            response = requests.get(
                f"{API_BASE_URL}/documents/{doc_id}/processed-image/{page_number}",
                timeout=30
            )
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            st.warning(f"Failed to fetch processed image: {e}")
            return None
    
    @staticmethod
    def get_extraction_with_layout(extraction_id: str) -> Dict[str, Any]:
        """
        Get extraction with layout data for bounding box overlay.
        Returns fields, layout_boxes, processed_image_paths, and page_dimensions.
        """
        try:
            response = requests.get(f"{API_BASE_URL}/extractions/{extraction_id}")
            data = APIClient._handle_response(response)
            
            # The extraction response already contains layout_data, processed_image_paths, etc.
            return data
        except Exception as e:
            return {"error": str(e), "success": False}

