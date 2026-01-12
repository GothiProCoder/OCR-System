"""
Export Schemas (Pydantic Models)
================================
Request/Response models for export functionality.

Supports:
    - Excel export (XLSX)
    - JSON export
    - PDF export (document + extracted data report)
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class ExportFormat(str, Enum):
    """Supported export formats"""
    EXCEL = "excel"
    JSON = "json"
    PDF = "pdf"
    CSV = "csv"


class ExportStatus(str, Enum):
    """Export job status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class ExportRequest(BaseModel):
    """Request to export extraction data"""
    
    extraction_id: UUID = Field(
        ...,
        description="Extraction to export"
    )
    format: ExportFormat = Field(
        ExportFormat.EXCEL,
        description="Export format"
    )
    
    # Export Options
    include_original_image: bool = Field(
        False,
        description="Include original document image (PDF only)"
    )
    include_ocr_text: bool = Field(
        False,
        description="Include raw OCR text output"
    )
    include_metadata: bool = Field(
        True,
        description="Include document and extraction metadata"
    )
    include_confidence_scores: bool = Field(
        True,
        description="Include confidence scores for each field"
    )
    
    # Field Selection
    field_ids: Optional[List[UUID]] = Field(
        None,
        description="Specific fields to export (None = all)"
    )
    exclude_low_confidence: bool = Field(
        False,
        description="Exclude fields with confidence < 0.60"
    )
    
    # Formatting Options
    date_format: str = Field(
        "%Y-%m-%d",
        description="Date format for date fields"
    )
    custom_filename: Optional[str] = Field(
        None,
        max_length=255,
        description="Custom filename without extension"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "extraction_id": "def12345-e89b-12d3-a456-426614174000",
                "format": "excel",
                "include_original_image": False,
                "include_metadata": True,
                "include_confidence_scores": True,
                "exclude_low_confidence": False,
                "date_format": "%Y-%m-%d",
                "custom_filename": "application_john_smith"
            }
        }
    )


class BulkExportRequest(BaseModel):
    """Request to export multiple extractions"""
    
    extraction_ids: List[UUID] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of extraction IDs to export"
    )
    format: ExportFormat = Field(
        ExportFormat.EXCEL,
        description="Export format"
    )
    
    # Options
    merge_into_single_file: bool = Field(
        True,
        description="Merge all into single file (Excel: multiple sheets)"
    )
    include_summary_sheet: bool = Field(
        True,
        description="Include summary sheet (Excel only)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "extraction_ids": [
                    "id-1-here",
                    "id-2-here",
                    "id-3-here"
                ],
                "format": "excel",
                "merge_into_single_file": True,
                "include_summary_sheet": True
            }
        }
    )


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class ExportResponse(BaseModel):
    """Response after export is generated"""
    
    export_id: UUID = Field(..., description="Export job ID")
    extraction_id: UUID = Field(..., description="Source extraction ID")
    format: ExportFormat = Field(..., description="Export format")
    status: ExportStatus = Field(..., description="Export status")
    
    # File Info
    filename: str = Field(..., description="Generated filename")
    file_path: str = Field(..., description="Download path")
    file_size_bytes: Optional[int] = Field(None, description="File size")
    
    # Download
    download_url: str = Field(..., description="URL to download file")
    expires_at: Optional[datetime] = Field(
        None,
        description="URL expiration time"
    )
    
    # Metadata
    created_at: datetime
    message: str = Field("Export generated successfully")
    
    @property
    def file_size_human(self) -> Optional[str]:
        if not self.file_size_bytes:
            return None
        size = self.file_size_bytes
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "export_id": "exp12345-e89b-12d3-a456-426614174000",
                "extraction_id": "def12345-e89b-12d3-a456-426614174000",
                "format": "excel",
                "status": "completed",
                "filename": "application_john_smith.xlsx",
                "file_path": "exports/2024/01/application_john_smith.xlsx",
                "file_size_bytes": 15360,
                "download_url": "/api/exports/exp12345.../download",
                "expires_at": "2024-01-15T12:00:00Z",
                "created_at": "2024-01-15T10:30:00Z",
                "message": "Export generated successfully"
            }
        }
    )


class BulkExportResponse(BaseModel):
    """Response for bulk export"""
    
    export_id: UUID = Field(..., description="Bulk export job ID")
    extraction_count: int = Field(..., description="Number of extractions")
    format: ExportFormat
    status: ExportStatus
    
    filename: str
    download_url: str
    file_size_bytes: Optional[int] = None
    
    created_at: datetime
    message: str = "Bulk export generated successfully"
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "export_id": "bulk-exp-12345",
                "extraction_count": 15,
                "format": "excel",
                "status": "completed",
                "filename": "bulk_export_2024-01-15.xlsx",
                "download_url": "/api/exports/bulk-exp-12345/download",
                "file_size_bytes": 51200,
                "created_at": "2024-01-15T10:30:00Z",
                "message": "Bulk export generated successfully"
            }
        }
    )


# =============================================================================
# EXPORT DATA STRUCTURE SCHEMAS
# =============================================================================

class ExportFieldData(BaseModel):
    """Field data in export"""
    
    key: str
    value: Optional[str]
    type: str
    confidence: Optional[float] = None
    confidence_percent: Optional[int] = None
    is_edited: bool = False
    page: int = 1


class ExportDocumentData(BaseModel):
    """Document data in export"""
    
    document_id: str
    filename: str
    form_type: Optional[str] = None
    language: Optional[str] = None
    page_count: int = 1
    uploaded_at: datetime
    finalized_at: Optional[datetime] = None


class ExportExtractionData(BaseModel):
    """Complete extraction data for export"""
    
    extraction_id: str
    version: int
    document: ExportDocumentData
    fields: List[ExportFieldData]
    
    # Statistics
    total_fields: int
    edited_fields: int
    avg_confidence: Optional[float] = None
    
    # Metadata
    created_at: datetime
    finalized_at: Optional[datetime] = None


class ExportSummary(BaseModel):
    """Summary for bulk exports"""
    
    total_extractions: int
    total_fields: int
    total_edited_fields: int
    avg_confidence: Optional[float] = None
    
    by_form_type: Dict[str, int] = Field(default_factory=dict)
    by_language: Dict[str, int] = Field(default_factory=dict)
    
    export_date: datetime
    generated_by: str = "FormExtract AI"
