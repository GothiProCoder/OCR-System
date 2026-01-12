"""
Schemas Package
===============
Pydantic models for API request/response validation.

Usage:
    from schemas import (
        # Document
        DocumentResponse, DocumentUploadMeta, DocumentListResponse,
        
        # Extraction
        ExtractionResponse, ExtractedFieldResponse, BulkFieldUpdate,
        
        # Export
        ExportRequest, ExportResponse, ExportFormat
    )
"""

# Document Schemas
from schemas.document import (
    # Enums
    DocumentStatus,
    FileType,
    
    # Request
    DocumentUploadMeta,
    DocumentUpdate,
    DocumentSearch,
    
    # Response
    DocumentResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentStats,
    DocumentUploadResponse,
    DocumentDeleteResponse,
)

# Extraction Schemas
from schemas.extraction import (
    # Enums
    FieldType,
    ConfidenceLevel,
    ExtractionStatus,
    
    # Field Schemas
    BoundingBox,
    ExtractedFieldCreate,
    ExtractedFieldResponse,
    ExtractedFieldUpdate,
    BulkFieldUpdate,
    
    # Extraction Schemas
    ExtractionRequest,
    ExtractionResponse,
    ExtractionListItem,
    ExtractionListResponse,
    ExtractionStartResponse,
    
    # Action Schemas
    FinalizationRequest,
    FinalizationResponse,
    FieldUpdateResponse,
    
    # Internal Schemas
    OCRResult,
    LLMExtractionResult,
)

# Export Schemas
from schemas.export import (
    # Enums
    ExportFormat,
    ExportStatus,
    
    # Request
    ExportRequest,
    BulkExportRequest,
    
    # Response
    ExportResponse,
    BulkExportResponse,
    
    # Data Structures
    ExportFieldData,
    ExportDocumentData,
    ExportExtractionData,
    ExportSummary,
)


__all__ = [
    # === Document ===
    "DocumentStatus",
    "FileType",
    "DocumentUploadMeta",
    "DocumentUpdate",
    "DocumentSearch",
    "DocumentResponse",
    "DocumentListItem",
    "DocumentListResponse",
    "DocumentStats",
    "DocumentUploadResponse",
    "DocumentDeleteResponse",
    
    # === Extraction ===
    "FieldType",
    "ConfidenceLevel",
    "ExtractionStatus",
    "BoundingBox",
    "ExtractedFieldCreate",
    "ExtractedFieldResponse",
    "ExtractedFieldUpdate",
    "BulkFieldUpdate",
    "ExtractionRequest",
    "ExtractionResponse",
    "ExtractionListItem",
    "ExtractionListResponse",
    "ExtractionStartResponse",
    "FinalizationRequest",
    "FinalizationResponse",
    "FieldUpdateResponse",
    "OCRResult",
    "LLMExtractionResult",
    
    # === Export ===
    "ExportFormat",
    "ExportStatus",
    "ExportRequest",
    "BulkExportRequest",
    "ExportResponse",
    "BulkExportResponse",
    "ExportFieldData",
    "ExportDocumentData",
    "ExportExtractionData",
    "ExportSummary",
]
