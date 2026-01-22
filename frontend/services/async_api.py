"""
Async HTTP Client Service
=========================
Async HTTP client using httpx for improved performance in the frontend.

Usage:
    from services.async_api import async_api
    
    # In async context
    async with async_api as client:
        docs = await client.list_documents()
"""

import os
import httpx
import streamlit as st
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager
import asyncio
import logging

logger = logging.getLogger(__name__)

# Base URL for Backend API - configurable via environment variable
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

# Timeout configuration
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
UPLOAD_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class AsyncAPIClient:
    """
    Async HTTP client for the OCR Backend API.
    
    Uses httpx for true async HTTP requests, improving performance
    when making multiple concurrent requests.
    
    Usage:
        async with AsyncAPIClient() as client:
            docs = await client.list_documents()
    """
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self) -> "AsyncAPIClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=DEFAULT_TIMEOUT,
            headers={"Accept": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager and close client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Process API response and handle errors."""
        try:
            if response.status_code in [200, 201]:
                return response.json()
            else:
                logger.warning(f"API Error ({response.status_code}): {response.text}")
                return {"error": response.text, "success": False}
        except ValueError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {"error": str(e), "success": False}
    
    # --- Document Endpoints ---
    
    async def upload_document(
        self, 
        file_content: bytes,
        filename: str,
        content_type: str,
        auto_extract: bool = True
    ) -> Dict[str, Any]:
        """Upload a file to the backend."""
        try:
            files = {"file": (filename, file_content, content_type)}
            data = {"auto_extract": str(auto_extract).lower()}
            
            response = await self._client.post(
                "/documents/upload",
                files=files,
                data=data,
                timeout=UPLOAD_TIMEOUT
            )
            return self._handle_response(response)
        except httpx.TimeoutException:
            return {"error": "Upload timed out. Please try again.", "success": False}
        except httpx.ConnectError:
            return {"error": "Cannot connect to API server.", "success": False}
        except httpx.RequestError as e:
            return {"error": f"Connection failed: {str(e)}", "success": False}
    
    async def list_documents(
        self, 
        limit: int = 50, 
        skip: int = 0
    ) -> Union[Dict[str, Any], List]:
        """Get list of uploaded documents."""
        try:
            response = await self._client.get(
                "/documents",
                params={"limit": limit, "skip": skip}
            )
            if response.status_code == 200:
                return response.json()
            return []
        except httpx.TimeoutException:
            logger.warning("Request timed out while fetching documents.")
            return []
        except httpx.ConnectError:
            logger.warning("Cannot connect to API server.")
            return []
        except httpx.RequestError as e:
            logger.warning(f"Failed to fetch documents: {e}")
            return []
    
    async def get_document_details(self, doc_id: str) -> Dict[str, Any]:
        """Get details of a specific document."""
        try:
            response = await self._client.get(f"/documents/{doc_id}")
            return self._handle_response(response)
        except httpx.TimeoutException:
            return {"error": "Request timed out", "success": False}
        except httpx.ConnectError:
            return {"error": "Cannot connect to API server", "success": False}
        except httpx.RequestError as e:
            return {"error": str(e), "success": False}
    
    async def get_extraction(self, extraction_id: str) -> Dict[str, Any]:
        """Get full extraction results including fields."""
        try:
            response = await self._client.get(f"/extractions/{extraction_id}")
            return self._handle_response(response)
        except httpx.TimeoutException:
            return {"error": "Request timed out", "success": False}
        except httpx.ConnectError:
            return {"error": "Cannot connect to API server", "success": False}
        except httpx.RequestError as e:
            return {"error": str(e), "success": False}
    
    async def update_field(
        self, 
        extraction_id: str, 
        field_id: str, 
        value: str
    ) -> Dict[str, Any]:
        """Update a specific field's value."""
        try:
            response = await self._client.patch(
                f"/extractions/{extraction_id}/fields/{field_id}",
                json={"field_value": value}
            )
            return self._handle_response(response)
        except httpx.TimeoutException:
            return {"error": "Request timed out", "success": False}
        except httpx.ConnectError:
            return {"error": "Cannot connect to API server", "success": False}
        except httpx.RequestError as e:
            return {"error": str(e), "success": False}
    
    async def get_dashboard_stats(self, period: str = "week") -> Dict[str, Any]:
        """Get dashboard statistics."""
        try:
            response = await self._client.get(
                "/stats/dashboard",
                params={"period": period}
            )
            return self._handle_response(response)
        except httpx.TimeoutException:
            return {"error": "Request timed out", "success": False}
        except httpx.ConnectError:
            return {"error": "Cannot connect to API server", "success": False}
        except httpx.RequestError as e:
            return {"error": str(e), "success": False}
    
    async def get_processed_image_bytes(
        self, 
        doc_id: str, 
        page_number: int = 1
    ) -> Optional[bytes]:
        """Fetch processed image bytes."""
        try:
            response = await self._client.get(
                f"/documents/{doc_id}/processed-image/{page_number}"
            )
            if response.status_code == 200:
                return response.content
            return None
        except httpx.TimeoutException:
            logger.warning("Request timed out while fetching image.")
            return None
        except httpx.ConnectError:
            logger.warning("Cannot connect to API server.")
            return None
        except httpx.RequestError as e:
            logger.warning(f"Failed to fetch processed image: {e}")
            return None


# Convenience function to run async code in Streamlit
def run_async(coro):
    """
    Run an async coroutine from sync Streamlit code.
    
    Uses asyncio.run() which is the recommended pattern for Python 3.7+.
    For nested event loops (e.g., in Jupyter), consider using nest_asyncio.
    
    Usage:
        result = run_async(client.list_documents())
    """
    # asyncio.run() is the modern, recommended approach (Python 3.7+)
    # It properly creates, runs, and closes the event loop
    return asyncio.run(coro)


# Default async client instance
async_api = AsyncAPIClient()
