"""
Extraction Schemas (Pydantic Models)
====================================
Request/Response models for extraction-related API endpoints.

Covers:
    - OCR processing requests/responses
    - Extracted field management
    - Bulk field updates
    - Finalization workflow
    - Version management
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict, computed_field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class FieldType(str, Enum):
    """Types of extracted fields"""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    EMAIL = "email"
    PHONE = "phone"
    CHECKBOX = "checkbox"
    TABLE = "table"
    SIGNATURE = "signature"
    ADDRESS = "address"
    NAME = "name"
    CURRENCY = "currency"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence level categories"""
    HIGH = "high"       # >= 0.85
    MEDIUM = "medium"   # >= 0.60
    LOW = "low"         # < 0.60


class ExtractionStatus(str, Enum):
    """Extraction processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# BOUNDING BOX SCHEMA
# =============================================================================

class BoundingBox(BaseModel):
    """Position of extracted field on document"""
    
    x: float = Field(..., ge=0, description="X coordinate (pixels or %)")
    y: float = Field(..., ge=0, description="Y coordinate (pixels or %)")
    width: float = Field(..., gt=0, description="Width")
    height: float = Field(..., gt=0, description="Height")
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    unit: str = Field("pixel", description="Unit type: 'pixel' or 'percent'")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "x": 120.5,
                "y": 340.0,
                "width": 200.0,
                "height": 25.0,
                "page": 1,
                "unit": "pixel"
            }
        }
    )


# =============================================================================
# EXTRACTED FIELD SCHEMAS
# =============================================================================

class ExtractedFieldBase(BaseModel):
    """Base schema for extracted fields"""
    
    field_key: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Field label/name"
    )
    field_value: Optional[str] = Field(
        None,
        description="Extracted value"
    )
    field_type: FieldType = Field(
        FieldType.TEXT,
        description="Detected field type"
    )
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )


class ExtractedFieldCreate(ExtractedFieldBase):
    """Schema for creating extracted fields (internal use)"""
    
    bounding_box: Optional[BoundingBox] = None
    page_number: int = Field(1, ge=1)
    sort_order: int = Field(0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedFieldResponse(ExtractedFieldBase):
    """Full extracted field response"""
    
    id: UUID = Field(..., description="Field ID")
    extraction_id: UUID = Field(..., description="Parent extraction ID")
    
    is_valid: bool = Field(True, description="Validation status")
    validation_message: Optional[str] = Field(None, description="Validation message")
    
    is_edited: bool = Field(False, description="Was manually edited")
    original_value: Optional[str] = Field(None, description="Value before editing")
    
    bounding_box: Optional[BoundingBox] = None
    page_number: int = Field(1)
    sort_order: int = Field(0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    created_at: datetime
    updated_at: datetime
    
    @computed_field
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Categorized confidence level"""
        if self.confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
    
    @computed_field
    @property
    def confidence_percent(self) -> int:
        """Confidence as percentage"""
        return int(self.confidence * 100)
    
    @computed_field
    @property
    def confidence_icon(self) -> str:
        """Emoji indicator for confidence"""
        if self.confidence >= 0.85:
            return "🟢"
        elif self.confidence >= 0.60:
            return "🟡"
        return "🔴"
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "abc12345-e89b-12d3-a456-426614174000",
                "extraction_id": "def12345-e89b-12d3-a456-426614174000",
                "field_key": "Full Name",
                "field_value": "John Smith",
                "field_type": "name",
                "confidence": 0.95,
                "is_valid": True,
                "validation_message": None,
                "is_edited": False,
                "original_value": None,
                "bounding_box": {
                    "x": 120, "y": 340, "width": 200, "height": 25, "page": 1
                },
                "page_number": 1,
                "sort_order": 0,
                "metadata": {},
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }
    )


class ExtractedFieldUpdate(BaseModel):
    """Schema for updating a single extracted field"""
    
    field_value: Optional[str] = Field(
        None,
        description="New field value"
    )
    field_type: Optional[FieldType] = Field(
        None,
        description="Update field type"
    )
    is_valid: Optional[bool] = Field(
        None,
        description="Manual validation override"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Update metadata"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "field_value": "Jane Smith",
                "is_valid": True
            }
        }
    )


class BulkFieldUpdate(BaseModel):
    """Schema for updating multiple fields at once"""
    
    updates: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of field updates with 'id' and fields to update"
    )
    
    @field_validator('updates')
    @classmethod
    def validate_updates(cls, v):
        for update in v:
            if 'id' not in update:
                raise ValueError("Each update must have an 'id' field")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "updates": [
                    {"id": "uuid-1", "field_value": "John Smith"},
                    {"id": "uuid-2", "field_value": "john@example.com"},
                    {"id": "uuid-3", "field_value": "2024-01-15"}
                ]
            }
        }
    )


# =============================================================================
# EXTRACTION SCHEMAS
# =============================================================================

class ExtractionRequest(BaseModel):
    """Request to start extraction for a document"""
    
    document_id: UUID = Field(..., description="Document to process")
    form_template_id: Optional[UUID] = Field(
        None,
        description="Template to apply (optional)"
    )
    custom_prompt: Optional[str] = Field(
        None,
        max_length=2000,
        description="Custom extraction prompt for LLM"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "form_template_id": None,
                "custom_prompt": None
            }
        }
    )


class ExtractionResponse(BaseModel):
    """Full extraction response with fields"""
    
    id: UUID = Field(..., description="Extraction ID")
    document_id: UUID = Field(..., description="Source document ID")
    version: int = Field(..., description="Version number")
    is_current: bool = Field(..., description="Is current version")
    
    status: ExtractionStatus = Field(..., description="Processing status")
    error_message: Optional[str] = Field(None, description="Error if failed")
    
    # OCR Output
    raw_ocr_markdown: Optional[str] = Field(None, description="Raw OCR text")
    
    # LLM Results
    form_type: Optional[str] = Field(None, description="Detected form type")
    language: Optional[str] = Field(None, description="Detected language")
    
    # Statistics
    confidence_avg: Optional[float] = Field(None, description="Avg confidence")
    total_fields: int = Field(0, description="Total field count")
    edited_fields_count: int = Field(0, description="Edited field count")
    
    # Processing Times
    processing_time_ms: Optional[int] = Field(None, description="Total time")
    ocr_time_ms: Optional[int] = Field(None, description="OCR time")
    llm_time_ms: Optional[int] = Field(None, description="LLM time")
    
    # Finalization
    is_finalized: bool = Field(False, description="Is finalized")
    finalized_at: Optional[datetime] = Field(None, description="Finalization time")
    
    # Extracted Fields
    fields: List[ExtractedFieldResponse] = Field(
        default_factory=list,
        description="Extracted key-value pairs"
    )
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    
    @computed_field
    @property
    def confidence_avg_percent(self) -> Optional[int]:
        """Average confidence as percentage"""
        if self.confidence_avg is None:
            return None
        return int(self.confidence_avg * 100)
    
    @computed_field
    @property
    def high_confidence_count(self) -> int:
        """Count of high confidence fields"""
        return sum(1 for f in self.fields if f.confidence >= 0.85)
    
    @computed_field
    @property
    def low_confidence_count(self) -> int:
        """Count of low confidence fields requiring review"""
        return sum(1 for f in self.fields if f.confidence < 0.60)
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "def12345-e89b-12d3-a456-426614174000",
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "version": 1,
                "is_current": True,
                "status": "completed",
                "error_message": None,
                "raw_ocr_markdown": "# Application Form\n\nName: John Smith...",
                "form_type": "Application Form",
                "language": "en",
                "confidence_avg": 0.89,
                "total_fields": 12,
                "edited_fields_count": 2,
                "processing_time_ms": 3500,
                "ocr_time_ms": 2100,
                "llm_time_ms": 1400,
                "is_finalized": False,
                "finalized_at": None,
                "fields": [],
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z"
            }
        }
    )


class ExtractionListItem(BaseModel):
    """Lightweight extraction for list views"""
    
    id: UUID
    document_id: UUID
    version: int
    is_current: bool
    status: ExtractionStatus
    form_type: Optional[str] = None
    confidence_avg: Optional[float] = None
    total_fields: int = 0
    is_finalized: bool = False
    created_at: datetime
    
    @computed_field
    @property
    def confidence_avg_percent(self) -> Optional[int]:
        if self.confidence_avg is None:
            return None
        return int(self.confidence_avg * 100)
    
    model_config = ConfigDict(from_attributes=True)


class ExtractionListResponse(BaseModel):
    """Paginated list of extractions"""
    
    items: List[ExtractionListItem]
    total: int
    page: int
    page_size: int


# =============================================================================
# ACTION SCHEMAS
# =============================================================================

class ExtractionStartResponse(BaseModel):
    """Response when extraction is started"""
    
    extraction_id: UUID = Field(..., description="New extraction ID")
    document_id: UUID = Field(..., description="Source document")
    status: ExtractionStatus = Field(..., description="Initial status")
    message: str = Field(..., description="Status message")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "extraction_id": "def12345-e89b-12d3-a456-426614174000",
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "processing",
                "message": "Extraction started successfully"
            }
        }
    )


class FinalizationRequest(BaseModel):
    """Request to finalize extraction (save to DB permanently)"""
    
    confirm: bool = Field(
        ...,
        description="Confirm finalization (must be true)"
    )
    notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional notes about the extraction"
    )
    
    @field_validator('confirm')
    @classmethod
    def must_confirm(cls, v):
        if not v:
            raise ValueError("Must set confirm=true to finalize")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "confirm": True,
                "notes": "Reviewed and approved by John"
            }
        }
    )


class FinalizationResponse(BaseModel):
    """Response after finalizing extraction"""
    
    extraction_id: UUID
    is_finalized: bool = True
    finalized_at: datetime
    message: str = "Extraction finalized successfully"
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "extraction_id": "def12345-e89b-12d3-a456-426614174000",
                "is_finalized": True,
                "finalized_at": "2024-01-15T11:00:00Z",
                "message": "Extraction finalized successfully"
            }
        }
    )


class FieldUpdateResponse(BaseModel):
    """Response after updating field(s)"""
    
    updated_count: int = Field(..., description="Number of fields updated")
    fields: List[ExtractedFieldResponse] = Field(
        default_factory=list,
        description="Updated fields"
    )
    message: str = Field("Fields updated successfully")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "updated_count": 3,
                "fields": [],
                "message": "3 fields updated successfully"
            }
        }
    )


# =============================================================================
# OCR/LLM RAW RESULT SCHEMAS (Internal Use)
# =============================================================================

class OCRResult(BaseModel):
    """Raw OCR result from Chandra"""
    
    markdown: str = Field(..., description="Markdown output")
    html: str = Field(..., description="HTML output")
    json_output: Dict[str, Any] = Field(default_factory=dict, description="Structured JSON")
    processing_time_ms: int = Field(..., description="OCR processing time")
    success: bool = Field(True)
    error: Optional[str] = None


class LLMExtractionResult(BaseModel):
    """Raw LLM extraction result"""
    
    fields: List[ExtractedFieldCreate] = Field(default_factory=list)
    form_type: str = Field("Unknown")
    language: str = Field("en")
    raw_response: Optional[str] = None
    processing_time_ms: int = 0
    success: bool = True
    error: Optional[str] = None
