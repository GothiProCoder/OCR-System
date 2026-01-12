"""
Database Models (SQLAlchemy ORM)
================================
Comprehensive schema design for the OCR Form Extraction System.

Tables:
    - Document: Uploaded files metadata
    - Extraction: OCR + LLM extraction results (versioned)
    - ExtractedField: Individual key-value pairs
    - FieldEdit: Audit trail for field modifications
    - FormTemplate: Reusable extraction templates
    - ProcessingLog: Step-by-step processing audit

Design Principles:
    - UUID primary keys for distributed compatibility
    - Soft deletes where appropriate
    - JSONB for flexible/nested data
    - Timestamps on all tables
    - Proper indexing for query performance
    - Enum types for constrained values
"""

from sqlalchemy import (
    Column, String, Integer, BigInteger, Float, Boolean, Text, 
    DateTime, ForeignKey, Enum, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import enum

from database.connection import Base


# =============================================================================
# ENUM DEFINITIONS
# =============================================================================

class DocumentStatus(str, enum.Enum):
    """Status of document processing"""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(str, enum.Enum):
    """Supported file types"""
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    PDF = "pdf"


class FieldType(str, enum.Enum):
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


class EditType(str, enum.Enum):
    """Types of field edits"""
    MANUAL = "manual"
    SUGGESTION_ACCEPTED = "suggestion_accepted"
    AUTO_CORRECTION = "auto_correction"
    VALIDATION_FIX = "validation_fix"


class ProcessingStep(str, enum.Enum):
    """Processing pipeline steps"""
    UPLOAD = "upload"
    PREPROCESSING = "preprocessing"
    OCR = "ocr"
    LLM_EXTRACTION = "llm_extraction"
    VALIDATION = "validation"
    FINALIZATION = "finalization"


class LogStatus(str, enum.Enum):
    """Status of processing steps"""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_uuid():
    """Generate a new UUID string"""
    return str(uuid.uuid4())


# =============================================================================
# MODELS
# =============================================================================

class Document(Base):
    """
    Uploaded document metadata.
    
    Stores information about uploaded files including path, status,
    and processing metadata. Supports soft deletion.
    """
    __tablename__ = "documents"
    
    # Primary Key
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4,
        comment="Unique document identifier"
    )
    
    # File Information
    filename = Column(
        String(255), 
        nullable=False,
        comment="Stored filename (may be sanitized)"
    )
    original_filename = Column(
        String(255), 
        nullable=False,
        comment="Original uploaded filename"
    )
    file_path = Column(
        String(500), 
        nullable=False,
        comment="Relative path in storage"
    )
    file_type = Column(
        Enum(FileType, name="file_type_enum"),
        nullable=False,
        comment="File extension type"
    )
    mime_type = Column(
        String(100),
        comment="MIME type of the file"
    )
    file_size_bytes = Column(
        BigInteger,
        comment="File size in bytes"
    )
    page_count = Column(
        Integer, 
        default=1,
        comment="Number of pages (for PDFs)"
    )
    
    # Processing Status
    status = Column(
        Enum(DocumentStatus, name="document_status_enum"),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
        comment="Current processing status"
    )
    
    # Detected Information (populated after processing)
    form_type = Column(
        String(100),
        comment="Detected form category (e.g., Invoice, Application)"
    )
    language = Column(
        String(10),
        comment="Detected language code (e.g., en, hi)"
    )
    
    # Flexible Metadata
    metadata = Column(
        JSONB,
        default=dict,
        comment="Additional metadata (dimensions, color depth, etc.)"
    )
    
    # Soft Delete
    is_deleted = Column(
        Boolean, 
        default=False,
        index=True,
        comment="Soft delete flag"
    )
    deleted_at = Column(
        DateTime(timezone=True),
        comment="Deletion timestamp"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False,
        comment="Upload timestamp"
    )
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Last update timestamp"
    )
    
    # Relationships
    extractions = relationship(
        "Extraction",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="desc(Extraction.version)"
    )
    processing_logs = relationship(
        "ProcessingLog",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ProcessingLog.created_at"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_documents_status_created", "status", "created_at"),
        Index("idx_documents_form_type", "form_type"),
        Index("idx_documents_not_deleted", "is_deleted", postgresql_where=(is_deleted == False)),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status={self.status})>"
    
    @property
    def current_extraction(self):
        """Get the current (latest) extraction for this document"""
        for ext in self.extractions:
            if ext.is_current:
                return ext
        return None


class Extraction(Base):
    """
    OCR and LLM extraction results.
    
    Supports versioning - each edit creates a new version.
    Only one extraction per document can be marked as current.
    """
    __tablename__ = "extractions"
    
    # Primary Key
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Foreign Key
    document_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Versioning
    version = Column(
        Integer, 
        default=1,
        nullable=False,
        comment="Extraction version number"
    )
    is_current = Column(
        Boolean, 
        default=True,
        index=True,
        comment="Is this the current extraction version"
    )
    
    # Processing Status
    status = Column(
        Enum(DocumentStatus, name="extraction_status_enum", create_type=False),
        default=DocumentStatus.PENDING,
        nullable=False
    )
    error_message = Column(
        Text,
        comment="Error message if processing failed"
    )
    
    # Raw OCR Output (from Chandra)
    raw_ocr_markdown = Column(
        Text,
        comment="Full OCR output in Markdown format"
    )
    raw_ocr_html = Column(
        Text,
        comment="Full OCR output in HTML format"
    )
    raw_ocr_json = Column(
        JSONB,
        comment="Structured OCR output with layout info"
    )
    
    # LLM Extraction Results
    form_type = Column(
        String(100),
        comment="LLM-detected form type"
    )
    language = Column(
        String(10),
        comment="Detected document language"
    )
    llm_raw_response = Column(
        Text,
        comment="Raw LLM response for debugging"
    )
    
    # Statistics
    confidence_avg = Column(
        Float,
        CheckConstraint("confidence_avg >= 0 AND confidence_avg <= 1"),
        comment="Average confidence score (0-1)"
    )
    total_fields = Column(
        Integer, 
        default=0,
        comment="Total number of extracted fields"
    )
    edited_fields_count = Column(
        Integer, 
        default=0,
        comment="Number of manually edited fields"
    )
    
    # Processing Times
    processing_time_ms = Column(
        Integer,
        comment="Total processing time in milliseconds"
    )
    ocr_time_ms = Column(
        Integer,
        comment="OCR processing time in milliseconds"
    )
    llm_time_ms = Column(
        Integer,
        comment="LLM extraction time in milliseconds"
    )
    
    # Finalization
    is_finalized = Column(
        Boolean, 
        default=False,
        index=True,
        comment="Has user finalized this extraction"
    )
    finalized_at = Column(
        DateTime(timezone=True),
        comment="When the extraction was finalized"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    document = relationship("Document", back_populates="extractions")
    fields = relationship(
        "ExtractedField",
        back_populates="extraction",
        cascade="all, delete-orphan",
        order_by="ExtractedField.sort_order"
    )
    field_edits = relationship(
        "FieldEdit",
        back_populates="extraction",
        cascade="all, delete-orphan"
    )
    processing_logs = relationship(
        "ProcessingLog",
        back_populates="extraction",
        cascade="all, delete-orphan"
    )
    
    # Indexes and Constraints
    __table_args__ = (
        Index("idx_extractions_document_version", "document_id", "version"),
        Index("idx_extractions_current", "document_id", "is_current", 
              postgresql_where=(is_current == True)),
        UniqueConstraint("document_id", "version", name="uq_extraction_document_version"),
    )
    
    def __repr__(self):
        return f"<Extraction(id={self.id}, document_id={self.document_id}, version={self.version})>"


class ExtractedField(Base):
    """
    Individual key-value pairs extracted from documents.
    
    Stores field labels, values, confidence scores, bounding boxes,
    and validation status.
    """
    __tablename__ = "extracted_fields"
    
    # Primary Key
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Foreign Key
    extraction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("extractions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Field Content
    field_key = Column(
        String(255), 
        nullable=False,
        comment="Field label/name"
    )
    field_value = Column(
        Text,
        comment="Extracted value"
    )
    field_type = Column(
        Enum(FieldType, name="field_type_enum"),
        default=FieldType.TEXT,
        comment="Detected field type"
    )
    
    # Confidence & Validation
    confidence = Column(
        Float,
        CheckConstraint("confidence >= 0 AND confidence <= 1"),
        comment="Extraction confidence score (0-1)"
    )
    is_valid = Column(
        Boolean, 
        default=True,
        comment="Did field pass validation"
    )
    validation_message = Column(
        String(500),
        comment="Validation error/warning message"
    )
    
    # Edit Tracking
    is_edited = Column(
        Boolean, 
        default=False,
        index=True,
        comment="Was this field manually edited"
    )
    original_value = Column(
        Text,
        comment="Original value before editing"
    )
    
    # Position Information
    bounding_box = Column(
        JSONB,
        comment="Position: {x, y, width, height, page}"
    )
    page_number = Column(
        Integer, 
        default=1,
        comment="Page number (1-indexed)"
    )
    sort_order = Column(
        Integer, 
        default=0,
        comment="Display order"
    )
    
    # Flexible Metadata
    metadata = Column(
        JSONB,
        default=dict,
        comment="Additional field metadata"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationships
    extraction = relationship("Extraction", back_populates="fields")
    edits = relationship(
        "FieldEdit",
        back_populates="field",
        cascade="all, delete-orphan",
        order_by="desc(FieldEdit.created_at)"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_fields_extraction_order", "extraction_id", "sort_order"),
        Index("idx_fields_key", "field_key"),
        Index("idx_fields_edited", "extraction_id", "is_edited"),
    )
    
    def __repr__(self):
        return f"<ExtractedField(id={self.id}, key='{self.field_key}', confidence={self.confidence})>"


class FieldEdit(Base):
    """
    Audit trail for field modifications.
    
    Records every change made to extracted fields for
    accountability and potential rollback.
    """
    __tablename__ = "field_edits"
    
    # Primary Key
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Foreign Keys
    field_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("extracted_fields.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    extraction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("extractions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Denormalized for query efficiency"
    )
    
    # Edit Details
    old_value = Column(Text, comment="Value before edit")
    new_value = Column(Text, comment="Value after edit")
    edit_type = Column(
        Enum(EditType, name="edit_type_enum"),
        default=EditType.MANUAL,
        comment="Type of edit"
    )
    edit_source = Column(
        String(100),
        default="user",
        comment="Who/what made the edit"
    )
    edit_reason = Column(
        String(500),
        comment="Optional reason for edit"
    )
    
    # Timestamp
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    field = relationship("ExtractedField", back_populates="edits")
    extraction = relationship("Extraction", back_populates="field_edits")
    
    # Indexes
    __table_args__ = (
        Index("idx_field_edits_created", "created_at"),
    )
    
    def __repr__(self):
        return f"<FieldEdit(id={self.id}, field_id={self.field_id}, type={self.edit_type})>"


class FormTemplate(Base):
    """
    Reusable extraction templates.
    
    Allows defining expected fields and custom prompts
    for specific form types (e.g., "Invoice", "Job Application").
    """
    __tablename__ = "form_templates"
    
    # Primary Key
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Template Info
    name = Column(
        String(100), 
        nullable=False,
        unique=True,
        comment="Template name"
    )
    description = Column(
        Text,
        comment="Template description"
    )
    
    # Expected Fields Definition
    expected_fields = Column(
        JSONB,
        default=list,
        comment="Array of expected field definitions"
    )
    # Example: [{"key": "Full Name", "type": "name", "required": true}, ...]
    
    # Custom Extraction Settings
    extraction_prompt = Column(
        Text,
        comment="Custom LLM prompt for this form type"
    )
    validation_rules = Column(
        JSONB,
        default=dict,
        comment="Field-specific validation rules"
    )
    # Example: {"email": {"pattern": "^[\\w.-]+@...", "required": true}}
    
    # Sample Document (optional)
    sample_document_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("documents.id", ondelete="SET NULL"),
        comment="Reference sample document"
    )
    
    # Usage Tracking
    is_active = Column(
        Boolean, 
        default=True,
        index=True
    )
    usage_count = Column(
        Integer, 
        default=0,
        comment="Number of times template was used"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Relationship
    sample_document = relationship("Document", foreign_keys=[sample_document_id])
    
    def __repr__(self):
        return f"<FormTemplate(id={self.id}, name='{self.name}')>"


class ProcessingLog(Base):
    """
    Detailed processing audit log.
    
    Tracks each step of the processing pipeline
    for debugging and monitoring.
    """
    __tablename__ = "processing_logs"
    
    # Primary Key
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    # Foreign Keys
    document_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    extraction_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("extractions.id", ondelete="CASCADE"),
        index=True,
        comment="May be null for pre-extraction steps"
    )
    
    # Log Details
    step = Column(
        Enum(ProcessingStep, name="processing_step_enum"),
        nullable=False,
        comment="Processing pipeline step"
    )
    status = Column(
        Enum(LogStatus, name="log_status_enum"),
        nullable=False,
        comment="Step status"
    )
    message = Column(
        Text,
        comment="Human-readable status message"
    )
    details = Column(
        JSONB,
        default=dict,
        comment="Additional details (errors, metrics, etc.)"
    )
    
    # Performance
    duration_ms = Column(
        Integer,
        comment="Step duration in milliseconds"
    )
    
    # Timestamp
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    document = relationship("Document", back_populates="processing_logs")
    extraction = relationship("Extraction", back_populates="processing_logs")
    
    # Indexes
    __table_args__ = (
        Index("idx_processing_logs_document_step", "document_id", "step"),
        Index("idx_processing_logs_created", "created_at"),
    )
    
    def __repr__(self):
        return f"<ProcessingLog(id={self.id}, step={self.step}, status={self.status})>"
