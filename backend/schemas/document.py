"""
Document Schemas (Pydantic Models)
==================================
Request/Response models for document-related API endpoints.

Design Principles:
    - Separate Create, Update, Response models for clarity
    - Proper validation with descriptive errors
    - Example values for OpenAPI documentation
    - Computed fields for derived data
    - Proper datetime handling with timezone awareness
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict, computed_field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


# =============================================================================
# ENUMS (Mirror database enums for API)
# =============================================================================

class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(str, Enum):
    """Supported file types"""
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    PDF = "pdf"


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class DocumentBase(BaseModel):
    """Base document fields shared across schemas"""
    
    model_config = ConfigDict(
        from_attributes=True,  # Enable ORM mode
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "filename": "application_form.pdf",
                "form_type": "Application Form",
                "language": "en"
            }
        }
    )


# =============================================================================
# REQUEST SCHEMAS (Client → Server)
# =============================================================================

class DocumentUploadMeta(BaseModel):
    """
    Optional metadata sent with file upload.
    File itself is sent as multipart form data.
    """
    
    custom_filename: Optional[str] = Field(
        None,
        max_length=255,
        description="Custom filename to use instead of original"
    )
    form_template_id: Optional[UUID] = Field(
        None,
        description="ID of form template to apply for extraction"
    )
    auto_extract: bool = Field(
        True,
        description="Automatically start extraction after upload"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional custom metadata"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "custom_filename": "employee_application_001",
                "auto_extract": True,
                "metadata": {"department": "HR", "batch_id": "2024-01"}
            }
        }
    )


class DocumentUpdate(BaseModel):
    """Schema for updating document metadata"""
    
    filename: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Update stored filename"
    )
    form_type: Optional[str] = Field(
        None,
        max_length=100,
        description="Manually set form type"
    )
    language: Optional[str] = Field(
        None,
        max_length=10,
        pattern=r"^[a-z]{2}(-[A-Z]{2})?$",
        description="Language code (e.g., 'en', 'en-US')"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Update custom metadata"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "form_type": "Invoice",
                "language": "en"
            }
        }
    )


class DocumentSearch(BaseModel):
    """Schema for document search/filter parameters"""
    
    query: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Search in filename and form type"
    )
    status: Optional[DocumentStatus] = Field(
        None,
        description="Filter by processing status"
    )
    form_type: Optional[str] = Field(
        None,
        description="Filter by form type"
    )
    date_from: Optional[datetime] = Field(
        None,
        description="Filter documents created after this date"
    )
    date_to: Optional[datetime] = Field(
        None,
        description="Filter documents created before this date"
    )
    page: int = Field(
        1,
        ge=1,
        description="Page number"
    )
    page_size: int = Field(
        20,
        ge=1,
        le=100,
        description="Items per page"
    )
    
    @property
    def offset(self) -> int:
        """Calculate offset for database query"""
        return (self.page - 1) * self.page_size


# =============================================================================
# RESPONSE SCHEMAS (Server → Client)
# =============================================================================

class DocumentResponse(DocumentBase):
    """
    Full document response with all fields.
    Used for single document GET responses.
    """
    
    id: UUID = Field(..., description="Unique document ID")
    filename: str = Field(..., description="Stored filename")
    original_filename: str = Field(..., description="Original upload filename")
    file_path: str = Field(..., description="Storage path")
    file_type: FileType = Field(..., description="File extension type")
    mime_type: Optional[str] = Field(None, description="MIME type")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    page_count: int = Field(1, description="Number of pages")
    
    status: DocumentStatus = Field(..., description="Processing status")
    form_type: Optional[str] = Field(None, description="Detected form type")
    language: Optional[str] = Field(None, description="Detected language")
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    is_deleted: bool = Field(False, description="Soft delete flag")
    created_at: datetime = Field(..., description="Upload timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # Relationships (optional, included when requested)
    current_extraction_id: Optional[UUID] = Field(
        None, 
        description="ID of current extraction version"
    )
    extraction_count: Optional[int] = Field(
        None,
        description="Total number of extraction versions"
    )
    
    @computed_field
    @property
    def file_size_human(self) -> Optional[str]:
        """Human-readable file size"""
        if not self.file_size_bytes:
            return None
        
        size = self.file_size_bytes
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    @computed_field
    @property
    def status_display(self) -> str:
        """Display-friendly status"""
        return self.status.value.replace("_", " ").title()
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "application_form.pdf",
                "original_filename": "scan_001.pdf",
                "file_path": "uploads/2024/01/application_form.pdf",
                "file_type": "pdf",
                "mime_type": "application/pdf",
                "file_size_bytes": 1048576,
                "page_count": 3,
                "status": "completed",
                "form_type": "Application Form",
                "language": "en",
                "metadata": {},
                "is_deleted": False,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z",
                "current_extraction_id": "789e4567-e89b-12d3-a456-426614174000",
                "extraction_count": 2
            }
        }
    )


class DocumentListItem(BaseModel):
    """
    Lightweight document response for list views.
    Excludes heavy fields like metadata and paths.
    """
    
    id: UUID
    filename: str
    original_filename: str
    file_type: FileType
    file_size_bytes: Optional[int] = None
    page_count: int = 1
    status: DocumentStatus
    form_type: Optional[str] = None
    created_at: datetime
    
    @computed_field
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
    
    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Paginated list of documents"""
    
    items: List[DocumentListItem] = Field(..., description="List of documents")
    total: int = Field(..., description="Total count (all pages)")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")
    
    @computed_field
    @property
    def has_next(self) -> bool:
        return self.page < self.pages
    
    @computed_field
    @property
    def has_prev(self) -> bool:
        return self.page > 1
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [],
                "total": 42,
                "page": 1,
                "page_size": 20,
                "pages": 3
            }
        }
    )


class DocumentStats(BaseModel):
    """Document statistics for dashboard"""
    
    total_documents: int = Field(..., description="Total document count")
    documents_today: int = Field(0, description="Documents uploaded today")
    documents_this_week: int = Field(0, description="Documents uploaded this week")
    
    by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Count per status"
    )
    by_form_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count per form type (top 10)"
    )
    
    total_pages_processed: int = Field(0, description="Total pages across all docs")
    avg_processing_time_ms: Optional[float] = Field(
        None,
        description="Average processing time"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_documents": 156,
                "documents_today": 12,
                "documents_this_week": 45,
                "by_status": {
                    "completed": 140,
                    "processing": 5,
                    "failed": 11
                },
                "by_form_type": {
                    "Invoice": 78,
                    "Application Form": 45,
                    "Survey": 23
                },
                "total_pages_processed": 423,
                "avg_processing_time_ms": 2340.5
            }
        }
    )


# =============================================================================
# ACTION RESPONSE SCHEMAS
# =============================================================================

class DocumentUploadResponse(BaseModel):
    """Response after successful document upload"""
    
    id: UUID = Field(..., description="New document ID")
    filename: str = Field(..., description="Stored filename")
    status: DocumentStatus = Field(..., description="Initial status")
    message: str = Field(..., description="Status message")
    extraction_started: bool = Field(
        False,
        description="Whether extraction was auto-started"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "application_form.pdf",
                "status": "pending",
                "message": "Document uploaded successfully",
                "extraction_started": True
            }
        }
    )


class DocumentDeleteResponse(BaseModel):
    """Response after document deletion"""
    
    id: UUID = Field(..., description="Deleted document ID")
    deleted: bool = Field(..., description="Whether deletion succeeded")
    message: str = Field(..., description="Status message")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "deleted": True,
                "message": "Document deleted successfully"
            }
        }
    )
