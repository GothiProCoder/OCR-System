"""
Frontend API Client
===================
Client for interacting with the OCR Backend API.
Handles error catching, timeouts, and connection failures.
"""

import os
import requests
import streamlit as st
from typing import Dict, Any, Optional, List, Union

# Base URL for Backend API - configurable via environment variable
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

# Default timeout for API requests (seconds)
DEFAULT_TIMEOUT = 30
UPLOAD_TIMEOUT = 120  # Longer timeout for file uploads


class APIClient:
    """
    Client for interacting with the OCR Backend API.
    Handles error catching, session timeouts, and connection failures.
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
        except ValueError as e:
            st.error(f"Failed to parse JSON response: {str(e)}")
            return {"error": str(e), "success": False}
        except Exception as e:
            st.error(f"Unexpected error handling response: {str(e)}")
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
            response = requests.post(
                f"{API_BASE_URL}/documents/upload",
                files=files,
                data=data,
                timeout=UPLOAD_TIMEOUT
            )
            return APIClient._handle_response(response)
        except requests.Timeout:
            return {"error": "Upload timed out. Please try again.", "success": False}
        except requests.ConnectionError:
            return {"error": "Cannot connect to API server. Is the backend running?", "success": False}
        except requests.RequestException as e:
            return {"error": f"Connection failed: {str(e)}", "success": False}

    @staticmethod
    def list_documents(limit: int = 50, skip: int = 0) -> Union[Dict[str, Any], List]:
        """Get list of uploaded documents."""
        try:
            response = requests.get(
                f"{API_BASE_URL}/documents",
                params={"limit": limit, "skip": skip},
                timeout=DEFAULT_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
            return []
        except requests.Timeout:
            st.warning("Request timed out while fetching documents.")
            return []
        except requests.ConnectionError:
            st.warning("Cannot connect to API server.")
            return []
        except requests.RequestException as e:
            st.warning(f"Failed to fetch documents: {e}")
            return []

    # --- Extraction Endpoints ---
    
    @staticmethod
    def get_extraction_status(doc_id: str) -> Dict[str, Any]:
        """Get extraction status/result for a document."""
        try:
            response = requests.get(
                f"{API_BASE_URL}/documents/{doc_id}",
                timeout=DEFAULT_TIMEOUT
            )
            return APIClient._handle_response(response)
        except requests.Timeout:
            return {"status": "error", "error": "Request timed out"}
        except requests.ConnectionError:
            return {"status": "error", "error": "Cannot connect to API server"}
        except requests.RequestException as e:
            return {"status": "error", "error": str(e)}

    @staticmethod
    def get_document_details(doc_id: str) -> Dict[str, Any]:
        """Get details of a specific document."""
        try:
            response = requests.get(
                f"{API_BASE_URL}/documents/{doc_id}",
                timeout=DEFAULT_TIMEOUT
            )
            return APIClient._handle_response(response)
        except requests.Timeout:
            return {"error": "Request timed out", "success": False}
        except requests.ConnectionError:
            return {"error": "Cannot connect to API server", "success": False}
        except requests.RequestException as e:
            return {"error": str(e), "success": False}

    @staticmethod
    def get_extraction(extraction_id: str) -> Dict[str, Any]:
        """Get full extraction results including fields."""
        try:
            response = requests.get(
                f"{API_BASE_URL}/extractions/{extraction_id}",
                timeout=DEFAULT_TIMEOUT
            )
            return APIClient._handle_response(response)
        except requests.Timeout:
            return {"error": "Request timed out", "success": False}
        except requests.ConnectionError:
            return {"error": "Cannot connect to API server", "success": False}
        except requests.RequestException as e:
            return {"error": str(e), "success": False}

    @staticmethod
    def update_field(extraction_id: str, field_id: str, value: str) -> Dict[str, Any]:
        """Update a specific field's value."""
        try:
            data = {"field_value": value}
            response = requests.patch(
                f"{API_BASE_URL}/extractions/{extraction_id}/fields/{field_id}",
                json=data,
                timeout=DEFAULT_TIMEOUT
            )
            return APIClient._handle_response(response)
        except requests.Timeout:
            return {"error": "Request timed out", "success": False}
        except requests.ConnectionError:
            return {"error": "Cannot connect to API server", "success": False}
        except requests.RequestException as e:
            return {"error": str(e), "success": False}

    @staticmethod
    def get_dashboard_stats(period: str = "week") -> Dict[str, Any]:
        """Get comprehensive dashboard statistics."""
        try:
            response = requests.get(
                f"{API_BASE_URL}/stats/dashboard",
                params={"period": period},
                timeout=DEFAULT_TIMEOUT
            )
            return APIClient._handle_response(response)
        except requests.Timeout:
            return {"error": "Request timed out", "success": False}
        except requests.ConnectionError:
            return {"error": "Cannot connect to API server", "success": False}
        except requests.RequestException as e:
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
                timeout=DEFAULT_TIMEOUT
            )
            if response.status_code == 200:
                return response.content
            return None
        except requests.Timeout:
            st.warning("Request timed out while fetching image.")
            return None
        except requests.ConnectionError:
            st.warning("Cannot connect to API server.")
            return None
        except requests.RequestException as e:
            st.warning(f"Failed to fetch processed image: {e}")
            return None
    
    @staticmethod
    def get_extraction_with_layout(extraction_id: str) -> Dict[str, Any]:
        """
        Get extraction with layout data for bounding box overlay.
        Returns fields, layout_boxes, processed_image_paths, and page_dimensions.
        """
        try:
            response = requests.get(
                f"{API_BASE_URL}/extractions/{extraction_id}",
                timeout=DEFAULT_TIMEOUT
            )
            data = APIClient._handle_response(response)
            return data
        except requests.Timeout:
            return {"error": "Request timed out", "success": False}
        except requests.ConnectionError:
            return {"error": "Cannot connect to API server", "success": False}
        except requests.RequestException as e:
            return {"error": str(e), "success": False}
