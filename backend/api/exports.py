"""
Exports API Router
==================
Handles export of extracted data to various formats (Excel, JSON, CSV, PDF).

Endpoints:
    POST   /exports              - Create export
    GET    /exports/{id}         - Get export status/info
    GET    /exports/{id}/download - Download export file

Integration:
    - ExportService for multi-format export generation
    - ExtractionCRUD for fetching extraction data
    - FileManager for file operations
    - Pydantic schemas from schemas/export.py

Reference: FastAPI Knowledge Base
    - FileResponse for file downloads
    - StreamingResponse for large files
    - Path() for path parameters
    - HTTPException for error handling
"""

from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any
from uuid import UUID
from pathlib import Path as FilePath
from datetime import datetime
import logging

# Database
from database import (
    get_async_session,
    extraction_crud,
    Extraction,
)

# Schemas
from schemas.export import (
    ExportFormat,
    ExportStatus,
    ExportRequest,
    ExportResponse,
    BulkExportRequest,
    BulkExportResponse,
)

# Services
from services.export_service import ExportService, ExportResult

# Config
from config import settings

# Logger
logger = logging.getLogger(__name__)

# =============================================================================
# ROUTER INSTANCE
# =============================================================================

router = APIRouter()

# Service singleton
export_service = ExportService()

# In-memory export cache (in production, use Redis)
# Maps export_id -> ExportResult
_export_cache: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# DEPENDENCIES
# =============================================================================

async def get_extraction_or_404(
    extraction_id: UUID,
    db: AsyncSession = Depends(get_async_session)
) -> Extraction:
    """
    Dependency to get an extraction by ID or raise 404.
    """
    extraction = await extraction_crud.get_with_fields(db, extraction_id)
    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction with ID {extraction_id} not found"
        )
    return extraction


def get_export_or_404(export_id: UUID) -> Dict[str, Any]:
    """
    Get export info from cache or raise 404.
    
    Note: In production, this would query a database table.
    Currently uses in-memory cache for simplicity.
    """
    export_info = _export_cache.get(str(export_id))
    if not export_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export with ID {export_id} not found"
        )
    return export_info


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_content_type(format: ExportFormat) -> str:
    """Get MIME type for export format."""
    content_types = {
        ExportFormat.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ExportFormat.JSON: "application/json",
        ExportFormat.CSV: "text/csv",
        ExportFormat.PDF: "application/pdf"
    }
    return content_types.get(format, "application/octet-stream")


def result_to_response(result: ExportResult) -> ExportResponse:
    """Convert ExportResult to ExportResponse schema."""
    return ExportResponse(
        export_id=result.export_id,
        extraction_id=result.extraction_id,
        format=ExportFormat(result.format),
        status=ExportStatus(result.status),
        filename=result.file_name or f"export_{result.export_id}",
        file_path=result.file_path or "",
        file_size_bytes=result.file_size_bytes,
        download_url=result.download_url or f"/api/exports/{result.export_id}/download",
        expires_at=None,  # Could implement expiration
        created_at=datetime.utcnow(),
        message="Export generated successfully" if result.success else (result.error or "Export failed")
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post(
    "",
    response_model=ExportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create export",
    description="""
    Generate an export file from an extraction.
    
    **Supported formats:**
    - **Excel** (`.xlsx`) - Styled worksheet with metadata and fields
    - **JSON** - Structured JSON with all data
    - **CSV** - Simple flat format for spreadsheets
    - **PDF** - Professional report document
    
    **Options:**
    - Include/exclude metadata
    - Include/exclude confidence scores
    - Filter low-confidence fields
    - Custom filename
    """,
    responses={
        201: {"description": "Export created successfully"},
        400: {"description": "Invalid format or validation failed"},
        404: {"description": "Extraction not found"}
    }
)
async def create_export(
    request: ExportRequest,
    db: AsyncSession = Depends(get_async_session)
):
    """
    Create an export from an extraction.
    
    Validates the extraction exists, runs export service,
    and returns export metadata with download URL.
    
    Reference: FastAPI Knowledge Base - Section 3
        - Pydantic models for request body validation
        - response_model for automatic serialization
    """
    # Verify extraction exists
    extraction = await get_extraction_or_404(request.extraction_id, db)
    
    # Prepare export options
    options = {
        "include_metadata": request.include_metadata,
        "include_confidence": request.include_confidence_scores,
        "include_ocr_text": request.include_ocr_text,
        "exclude_low_confidence": request.exclude_low_confidence,
        "date_format": request.date_format,
        "field_ids": [str(fid) for fid in request.field_ids] if request.field_ids else None
    }
    
    # Run export
    try:
        result = await export_service.export_extraction(
            extraction_id=request.extraction_id,
            format=request.format.value,
            options=options,
            validate_first=False,  # Already validated if finalized
            custom_filename=request.custom_filename
        )
    except Exception as e:
        logger.error(f"Export failed for extraction {request.extraction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export generation failed: {str(e)}"
        )
    
    # Check result
    if not result.success:
        # Check if it's a validation failure
        if result.status == "validation_failed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": result.error,
                    "validation_errors": result.validation_errors
                }
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error or "Export generation failed"
        )
    
    # Cache export info for later retrieval
    _export_cache[str(result.export_id)] = result.to_dict()
    
    logger.info(f"Export created: {result.export_id} ({request.format.value}) for extraction {request.extraction_id}")
    
    return result_to_response(result)


@router.get(
    "/{export_id}",
    summary="Get export info",
    description="""
    Get information about an export.
    
    Returns status, file info, and download URL.
    """,
    responses={
        200: {"description": "Export info"},
        404: {"description": "Export not found"}
    }
)
async def get_export(
    export_id: UUID = Path(..., description="Export ID")
):
    """
    Get export status and info.
    
    Reference: FastAPI Knowledge Base - Section 2
        - Path() for path parameters
    """
    export_info = get_export_or_404(export_id)
    
    return {
        "export_id": export_info.get("export_id"),
        "extraction_id": export_info.get("extraction_id"),
        "format": export_info.get("format"),
        "status": export_info.get("status"),
        "filename": export_info.get("file_name"),
        "file_size_bytes": export_info.get("file_size_bytes"),
        "download_url": export_info.get("download_url"),
        "processing_time_ms": export_info.get("processing_time_ms"),
        "field_count": export_info.get("field_count"),
        "success": export_info.get("success"),
        "error": export_info.get("error")
    }


@router.get(
    "/{export_id}/download",
    summary="Download export file",
    description="""
    Download the generated export file.
    
    Returns the file as a binary download with appropriate
    content-type headers.
    """,
    responses={
        200: {"description": "File download"},
        404: {"description": "Export or file not found"}
    }
)
async def download_export(
    export_id: UUID = Path(..., description="Export ID")
):
    """
    Download export file.
    
    Uses FileResponse for efficient file streaming.
    
    Reference: FastAPI Knowledge Base
        - FileResponse for file downloads
        - Proper content-type headers
    """
    # Get export info
    export_info = get_export_or_404(export_id)
    
    file_path = export_info.get("file_path")
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export file path not found"
        )
    
    # Check file exists
    path = FilePath(file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export file not found on disk"
        )
    
    # Determine content type
    format_str = export_info.get("format", "excel")
    try:
        export_format = ExportFormat(format_str)
    except ValueError:
        export_format = ExportFormat.EXCEL
    
    content_type = get_content_type(export_format)
    filename = export_info.get("file_name", f"export_{export_id}.xlsx")
    
    logger.info(f"Downloading export: {export_id} ({filename})")
    
    return FileResponse(
        path=str(path),
        media_type=content_type,
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


# =============================================================================
# QUICK EXPORT ENDPOINTS (Direct download without creating export record)
# =============================================================================

@router.get(
    "/extraction/{extraction_id}/excel",
    summary="Quick export to Excel",
    description="Generate and download Excel file directly.",
    responses={
        200: {"description": "Excel file download"},
        404: {"description": "Extraction not found"}
    }
)
async def quick_export_excel(
    extraction_id: UUID = Path(..., description="Extraction ID"),
    include_metadata: bool = Query(True, description="Include metadata sheet"),
    include_confidence: bool = Query(True, description="Include confidence scores"),
    db: AsyncSession = Depends(get_async_session)
):
    """Quick export to Excel without creating an export record."""
    return await _quick_export(
        extraction_id=extraction_id,
        format=ExportFormat.EXCEL,
        include_metadata=include_metadata,
        include_confidence=include_confidence,
        db=db
    )


@router.get(
    "/extraction/{extraction_id}/json",
    summary="Quick export to JSON",
    description="Generate and download JSON file directly.",
    responses={
        200: {"description": "JSON file download"},
        404: {"description": "Extraction not found"}
    }
)
async def quick_export_json(
    extraction_id: UUID = Path(..., description="Extraction ID"),
    include_metadata: bool = Query(True, description="Include metadata"),
    db: AsyncSession = Depends(get_async_session)
):
    """Quick export to JSON without creating an export record."""
    return await _quick_export(
        extraction_id=extraction_id,
        format=ExportFormat.JSON,
        include_metadata=include_metadata,
        include_confidence=True,
        db=db
    )


@router.get(
    "/extraction/{extraction_id}/csv",
    summary="Quick export to CSV",
    description="Generate and download CSV file directly.",
    responses={
        200: {"description": "CSV file download"},
        404: {"description": "Extraction not found"}
    }
)
async def quick_export_csv(
    extraction_id: UUID = Path(..., description="Extraction ID"),
    include_confidence: bool = Query(False, description="Include confidence column"),
    db: AsyncSession = Depends(get_async_session)
):
    """Quick export to CSV without creating an export record."""
    return await _quick_export(
        extraction_id=extraction_id,
        format=ExportFormat.CSV,
        include_metadata=False,
        include_confidence=include_confidence,
        db=db
    )


@router.get(
    "/extraction/{extraction_id}/pdf",
    summary="Quick export to PDF",
    description="Generate and download PDF report directly.",
    responses={
        200: {"description": "PDF file download"},
        404: {"description": "Extraction not found"}
    }
)
async def quick_export_pdf(
    extraction_id: UUID = Path(..., description="Extraction ID"),
    include_metadata: bool = Query(True, description="Include document info"),
    db: AsyncSession = Depends(get_async_session)
):
    """Quick export to PDF without creating an export record."""
    return await _quick_export(
        extraction_id=extraction_id,
        format=ExportFormat.PDF,
        include_metadata=include_metadata,
        include_confidence=True,
        db=db
    )


async def _quick_export(
    extraction_id: UUID,
    format: ExportFormat,
    include_metadata: bool,
    include_confidence: bool,
    db: AsyncSession
) -> FileResponse:
    """
    Internal helper for quick export endpoints.
    
    Generates export file and returns FileResponse directly.
    """
    # Verify extraction exists
    extraction = await get_extraction_or_404(extraction_id, db)
    
    # Generate export
    options = {
        "include_metadata": include_metadata,
        "include_confidence": include_confidence
    }
    
    try:
        result = await export_service.export_extraction(
            extraction_id=extraction_id,
            format=format.value,
            options=options,
            validate_first=False
        )
    except Exception as e:
        logger.error(f"Quick export failed for {extraction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )
    
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error or "Export failed"
        )
    
    # Return file
    path = FilePath(result.file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Generated file not found"
        )
    
    content_type = get_content_type(format)
    filename = result.file_name or f"extraction_{extraction_id}{format.value}"
    
    return FileResponse(
        path=str(path),
        media_type=content_type,
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )
