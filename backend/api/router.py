"""
Central API Router
==================
Combines all sub-routers into a single API router with the /api prefix.

Usage in main.py:
    from api.router import api_router
    app.include_router(api_router, prefix="/api")

Router Structure:
    /api
    ├── /documents     - Document upload & management
    ├── /extractions   - OCR & extraction operations  
    ├── /exports       - Export functionality
    └── /stats         - Dashboard metrics

Reference: FastAPI Knowledge Base - Section 1 (APIRouter)
    - APIRouter allows grouping related path operations
    - Routers can have their own prefix, tags, dependencies, responses
    - Include routers using app.include_router()
"""

from fastapi import APIRouter

# =============================================================================
# MAIN API ROUTER
# =============================================================================

api_router = APIRouter()

# =============================================================================
# SUB-ROUTER IMPORTS & INCLUDES
# =============================================================================

# Each sub-router is conditionally imported to allow incremental development.
# As each router is implemented, uncomment its include statement.

# -----------------------------------------------------------------------------
# Documents Router - Document upload & management
# Endpoints:
#   POST   /documents/upload         - Upload document
#   GET    /documents                - List all documents
#   GET    /documents/{id}           - Get document details
#   DELETE /documents/{id}           - Delete document
# -----------------------------------------------------------------------------
try:
    from api.documents import router as documents_router
    api_router.include_router(
        documents_router,
        prefix="/documents",
        tags=["Documents"],
        responses={
            404: {"description": "Document not found"},
            400: {"description": "Invalid file type or size"}
        }
    )
except ImportError:
    pass  # Router not implemented yet

# -----------------------------------------------------------------------------
# Extractions Router - OCR & extraction operations
# Endpoints:
#   POST   /extractions/{document_id}           - Start extraction
#   GET    /extractions/{id}                    - Get extraction result
#   GET    /extractions/{id}/status             - Get extraction status
#   PATCH  /extractions/{id}/fields/{field_id}  - Update field
#   POST   /extractions/{id}/validate           - Run validation
#   POST   /extractions/{id}/finalize           - Finalize extraction
#   DELETE /extractions/{id}                    - Delete extraction
# -----------------------------------------------------------------------------
try:
    from api.extraction import router as extraction_router
    api_router.include_router(
        extraction_router,
        prefix="/extractions",
        tags=["Extractions"],
        responses={
            404: {"description": "Extraction not found"},
            409: {"description": "Extraction already finalized"}
        }
    )
except ImportError:
    pass  # Router not implemented yet

# -----------------------------------------------------------------------------
# Exports Router - Export functionality
# Endpoints:
#   POST   /exports                 - Create export
#   GET    /exports/{id}            - Get export status
#   GET    /exports/{id}/download   - Download export file
# -----------------------------------------------------------------------------
try:
    from api.exports import router as exports_router
    api_router.include_router(
        exports_router,
        prefix="/exports",
        tags=["Exports"],
        responses={
            404: {"description": "Export not found"},
            400: {"description": "Invalid export format"}
        }
    )
except ImportError:
    pass  # Router not implemented yet

# -----------------------------------------------------------------------------
# Stats Router - Dashboard metrics
# Endpoints:
#   GET    /stats/dashboard         - Dashboard metrics
# -----------------------------------------------------------------------------
try:
    from api.stats import router as stats_router
    api_router.include_router(
        stats_router,
        prefix="/stats",
        tags=["Statistics"]
    )
except ImportError:
    pass  # Router not implemented yet


# =============================================================================
# API INFO ENDPOINT (Always Available)
# =============================================================================

@api_router.get(
    "/",
    tags=["API Info"],
    summary="API Information",
    description="Returns information about the API and available endpoints."
)
async def api_info():
    """
    Returns API information and available endpoint groups.
    This endpoint is always available regardless of which routers are implemented.
    """
    return {
        "api_version": "v1",
        "title": "FormExtract AI API",
        "description": "OCR Form Data Extraction API",
        "endpoints": {
            "documents": {
                "prefix": "/api/documents",
                "description": "Document upload and management"
            },
            "extractions": {
                "prefix": "/api/extractions", 
                "description": "OCR processing and field extraction"
            },
            "exports": {
                "prefix": "/api/exports",
                "description": "Export extracted data in various formats"
            },
            "stats": {
                "prefix": "/api/stats",
                "description": "Dashboard metrics and analytics"
            }
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        }
    }
