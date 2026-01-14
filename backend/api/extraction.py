"""
Extraction API Router
=====================
Handles OCR extraction, field management, validation, and finalization.

Endpoints:
    POST   /extractions/{document_id}           - Start extraction
    GET    /extractions/{id}                    - Get extraction with fields
    GET    /extractions/{id}/status             - Get extraction status
    PATCH  /extractions/{id}/fields/{field_id}  - Update single field
    POST   /extractions/{id}/validate           - Run validation
    POST   /extractions/{id}/finalize           - Finalize extraction
    DELETE /extractions/{id}                    - Delete extraction

Integration:
    - ExtractionService for OCR + Gemini workflow orchestration
    - ValidationService for field validation
    - ExtractionCRUD / ExtractedFieldCRUD for database operations
    - Pydantic schemas from schemas/extraction.py

Reference: FastAPI Knowledge Base
    - Path() for path parameters (always required)
    - Depends() for dependency injection
    - BackgroundTasks for async processing
    - HTTPException for error handling
    - response_model for response serialization
"""

from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from uuid import UUID
from datetime import datetime
import logging

# Database
from database import (
    get_async_session,
    document_crud,
    extraction_crud,
    field_crud,
    Document,
    Extraction,
    ExtractedField,
    DocumentStatus,
)

# Schemas
from schemas.extraction import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionStartResponse,
    ExtractionStatus,
    ExtractedFieldResponse,
    ExtractedFieldUpdate,
    FieldUpdateResponse,
    FinalizationRequest,
    FinalizationResponse,
    BulkFieldUpdate,
)

# Services
from services.extraction_service import ExtractionService
from services.validation_service import ValidationService

# Config
from config import settings

# Logger
logger = logging.getLogger(__name__)

# =============================================================================
# ROUTER INSTANCE
# =============================================================================

router = APIRouter()

# Service singletons
extraction_service = ExtractionService()
validation_service = ValidationService()


# =============================================================================
# DEPENDENCIES
# =============================================================================

async def get_document_or_404(
    document_id: UUID = Path(..., description="Document ID"),
    db: AsyncSession = Depends(get_async_session)
) -> Document:
    """
    Dependency to get a document by ID or raise 404.
    Used for starting extraction.
    """
    document = await document_crud.get(db, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    if document.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} has been deleted"
        )
    return document


async def get_extraction_or_404(
    extraction_id: UUID = Path(..., alias="id", description="Extraction ID"),
    db: AsyncSession = Depends(get_async_session)
) -> Extraction:
    """
    Dependency to get an extraction by ID or raise 404.
    
    Reference: FastAPI Knowledge Base - Section 2
        - Depends() for nested dependencies
        - Path() for path parameters
    """
    extraction = await extraction_crud.get_with_fields(db, extraction_id)
    if not extraction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction with ID {extraction_id} not found"
        )
    return extraction


async def get_field_or_404(
    field_id: UUID = Path(..., description="Field ID"),
    db: AsyncSession = Depends(get_async_session)
) -> ExtractedField:
    """
    Dependency to get an extracted field by ID or raise 404.
    """
    field = await field_crud.get(db, field_id)
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field with ID {field_id} not found"
        )
    return field


def check_not_finalized(extraction: Extraction) -> None:
    """
    Raises 409 Conflict if extraction is already finalized.
    Finalized extractions cannot be modified.
    """
    if extraction.is_finalized:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Extraction {extraction.id} is finalized and cannot be modified"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extraction_to_response(extraction: Extraction) -> ExtractionResponse:
    """
    Convert Extraction ORM model to ExtractionResponse schema.
    """
    fields = [
        ExtractedFieldResponse(
            id=f.id,
            extraction_id=f.extraction_id,
            field_key=f.field_key,
            field_value=f.field_value,
            field_type=f.field_type.value if f.field_type else "text",
            confidence=f.confidence or 0.0,
            is_valid=f.is_valid if f.is_valid is not None else True,
            validation_message=f.validation_message,
            is_edited=f.is_edited or False,
            original_value=f.original_value,
            page_number=f.page_number or 1,
            created_at=f.created_at,
            updated_at=f.updated_at
        )
        for f in (extraction.fields or [])
    ]
    
    return ExtractionResponse(
        id=extraction.id,
        document_id=extraction.document_id,
        version=extraction.version,
        is_current=extraction.is_current,
        status=ExtractionStatus(extraction.status.value) if extraction.status else ExtractionStatus.PENDING,
        error_message=extraction.error_message,
        raw_ocr_markdown=extraction.raw_ocr_markdown,
        form_type=extraction.form_type,
        language=extraction.language,
        confidence_avg=extraction.confidence_avg,
        total_fields=extraction.total_fields or len(fields),
        edited_fields_count=extraction.edited_fields_count or 0,
        processing_time_ms=extraction.processing_time_ms,
        ocr_time_ms=extraction.ocr_time_ms,
        llm_time_ms=extraction.llm_time_ms,
        is_finalized=extraction.is_finalized or False,
        finalized_at=extraction.finalized_at,
        fields=fields,
        created_at=extraction.created_at,
        updated_at=extraction.updated_at
    )


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def run_extraction_background(
    document_id: UUID,
    file_path: str,
    file_type: str,
    extraction_id: UUID,
    custom_prompt: Optional[str] = None
):
    """
    Background task to run the extraction pipeline.
    
    Reference: FastAPI Knowledge Base - BackgroundTasks
        - Fire-and-forget operations after response is sent
        - Ideal for non-critical, lightweight operations
    """
    try:
        logger.info(f"Starting background extraction for document {document_id}")
        
        result = await extraction_service.extract_document(
            document_id=document_id,
            file_path=file_path,
            file_type=file_type,
            custom_prompt=custom_prompt
        )
        
        if result.success:
            logger.info(f"Extraction completed for document {document_id}: {result.total_fields} fields")
        else:
            logger.error(f"Extraction failed for document {document_id}: {result.error}")
            
    except Exception as e:
        logger.exception(f"Background extraction error for document {document_id}: {e}")


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post(
    "/{document_id}",
    response_model=ExtractionStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start extraction for a document",
    description="""
    Start the OCR + LLM extraction pipeline for a document.
    
    The extraction runs asynchronously. Poll the status endpoint
    to check progress and get results when complete.
    
    **Pipeline:**
    1. OCR - Extract text from document
    2. Gemini - Extract structured key-value pairs
    3. Validation - Validate extracted fields
    4. Database - Save extraction results
    """,
    responses={
        202: {"description": "Extraction started"},
        404: {"description": "Document not found"},
        409: {"description": "Extraction already in progress"}
    }
)
async def start_extraction(
    document: Document = Depends(get_document_or_404),
    custom_prompt: Optional[str] = Query(
        None,
        max_length=2000,
        description="Custom extraction instructions for the LLM"
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Start extraction for a document.
    
    Creates a new extraction version and kicks off background processing.
    Returns immediately with extraction ID for status polling.
    
    Reference: FastAPI Knowledge Base - Section 2
        - BackgroundTasks for async processing
        - Path() parameters are always required
    """
    # Check if document is already being processed
    if document.status == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document {document.id} is already being processed"
        )
    
    # Create new extraction record
    try:
        extraction = await extraction_crud.create_new_version(
            db,
            document_id=document.id,
            status=DocumentStatus.PROCESSING
        )
        await db.commit()
        await db.refresh(extraction)
    except Exception as e:
        logger.error(f"Failed to create extraction record: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create extraction record"
        )
    
    # Get file path from document
    file_path = str(settings.PROJECT_ROOT / document.file_path)
    file_type = document.file_type.value if document.file_type else "pdf"
    
    # Queue background extraction
    background_tasks.add_task(
        run_extraction_background,
        document_id=document.id,
        file_path=file_path,
        file_type=file_type,
        extraction_id=extraction.id,
        custom_prompt=custom_prompt
    )
    
    logger.info(f"Extraction started for document {document.id}: extraction_id={extraction.id}")
    
    return ExtractionStartResponse(
        extraction_id=extraction.id,
        document_id=document.id,
        status=ExtractionStatus.PROCESSING,
        message="Extraction started successfully. Poll status endpoint for progress."
    )


@router.get(
    "/{id}",
    response_model=ExtractionResponse,
    summary="Get extraction with fields",
    description="""
    Get detailed extraction information including all extracted fields.
    
    Includes OCR results, field data, confidence scores, and processing stats.
    """,
    responses={
        200: {"description": "Extraction details"},
        404: {"description": "Extraction not found"}
    }
)
async def get_extraction(
    extraction: Extraction = Depends(get_extraction_or_404)
):
    """
    Get a single extraction by ID with all fields.
    
    Reference: FastAPI Knowledge Base - Section 2
        - Depends() for reusable dependencies
        - response_model for automatic serialization
    """
    return extraction_to_response(extraction)


@router.get(
    "/{id}/status",
    summary="Get extraction status",
    description="""
    Get the current processing status of an extraction.
    
    Use this endpoint for polling during extraction processing.
    Returns lightweight status without full field data.
    """,
    responses={
        200: {"description": "Extraction status"},
        404: {"description": "Extraction not found"}
    }
)
async def get_extraction_status(
    extraction: Extraction = Depends(get_extraction_or_404)
):
    """
    Get extraction status for polling.
    
    Lightweight response for checking processing progress.
    """
    return {
        "id": extraction.id,
        "document_id": extraction.document_id,
        "status": extraction.status.value if extraction.status else "pending",
        "is_finalized": extraction.is_finalized or False,
        "total_fields": extraction.total_fields or 0,
        "confidence_avg": extraction.confidence_avg,
        "processing_time_ms": extraction.processing_time_ms,
        "error_message": extraction.error_message,
        "created_at": extraction.created_at,
        "updated_at": extraction.updated_at
    }


@router.patch(
    "/{id}/fields/{field_id}",
    response_model=FieldUpdateResponse,
    summary="Update a single field",
    description="""
    Update the value or metadata of a single extracted field.
    
    Changes are tracked in the audit log. Cannot update finalized extractions.
    """,
    responses={
        200: {"description": "Field updated"},
        404: {"description": "Extraction or field not found"},
        409: {"description": "Extraction is finalized"}
    }
)
async def update_field(
    update_data: ExtractedFieldUpdate,
    extraction: Extraction = Depends(get_extraction_or_404),
    field: ExtractedField = Depends(get_field_or_404),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Update a single extracted field.
    
    Creates an audit record for the change and marks field as edited.
    
    Reference: FastAPI Knowledge Base - Section 4
        - HTTPException for client errors (409 Conflict)
        - Proper error handling and rollback
    """
    # Check not finalized
    check_not_finalized(extraction)
    
    # Verify field belongs to this extraction
    if field.extraction_id != extraction.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Field {field.id} does not belong to extraction {extraction.id}"
        )
    
    try:
        # Update field value if provided
        if update_data.field_value is not None:
            updated_field = await field_crud.update_value(
                db,
                id=field.id,
                new_value=update_data.field_value,
                edit_source="api"
            )
        else:
            # Update other fields
            update_dict = update_data.model_dump(exclude_unset=True)
            if update_dict:
                updated_field = await field_crud.update(db, field.id, **update_dict)
            else:
                updated_field = field
        
        # Update extraction stats
        await extraction_crud.update_stats(db, extraction.id)
        
        await db.commit()
        await db.refresh(updated_field)
        
        # Build response
        field_response = ExtractedFieldResponse(
            id=updated_field.id,
            extraction_id=updated_field.extraction_id,
            field_key=updated_field.field_key,
            field_value=updated_field.field_value,
            field_type=updated_field.field_type.value if updated_field.field_type else "text",
            confidence=updated_field.confidence or 0.0,
            is_valid=updated_field.is_valid if updated_field.is_valid is not None else True,
            validation_message=updated_field.validation_message,
            is_edited=updated_field.is_edited or False,
            original_value=updated_field.original_value,
            page_number=updated_field.page_number or 1,
            created_at=updated_field.created_at,
            updated_at=updated_field.updated_at
        )
        
        logger.info(f"Field {field.id} updated in extraction {extraction.id}")
        
        return FieldUpdateResponse(
            updated_count=1,
            fields=[field_response],
            message="Field updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating field {field.id}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update field"
        )


@router.post(
    "/{id}/validate",
    summary="Run validation on extraction",
    description="""
    Run validation rules on all extracted fields.
    
    Checks field values against type-specific rules (email format,
    phone format, dates, etc.) and updates validation status in database.
    """,
    responses={
        200: {"description": "Validation results"},
        404: {"description": "Extraction not found"}
    }
)
async def validate_extraction(
    extraction: Extraction = Depends(get_extraction_or_404)
):
    """
    Run validation on extraction fields.
    
    Uses ValidationService to check all fields and update database.
    """
    try:
        result = await validation_service.validate_extraction(
            extraction_id=extraction.id,
            update_database=True
        )
        
        return {
            "extraction_id": extraction.id,
            "success": result.success,
            "total_fields": result.total_fields,
            "valid_count": result.valid_count,
            "invalid_count": result.invalid_count,
            "needs_review_count": result.needs_review_count,
            "overall_valid": result.overall_valid,
            "message": f"Validated {result.total_fields} fields: {result.valid_count} valid, {result.invalid_count} invalid"
        }
        
    except Exception as e:
        logger.error(f"Validation error for extraction {extraction.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Validation failed"
        )


@router.post(
    "/{id}/finalize",
    response_model=FinalizationResponse,
    summary="Finalize extraction",
    description="""
    Mark extraction as finalized after review.
    
    Finalized extractions cannot be modified. This is the final step
    before exporting data.
    
    Runs validation before finalizing to ensure data quality.
    """,
    responses={
        200: {"description": "Extraction finalized"},
        400: {"description": "Validation failed"},
        404: {"description": "Extraction not found"},
        409: {"description": "Already finalized"}
    }
)
async def finalize_extraction(
    request: FinalizationRequest,
    extraction: Extraction = Depends(get_extraction_or_404),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Finalize extraction after review.
    
    Runs validation, checks for critical errors, and marks as finalized.
    """
    # Check not already finalized
    check_not_finalized(extraction)
    
    # Run validation
    validation_result = await validation_service.validate_extraction(
        extraction_id=extraction.id,
        update_database=True
    )
    
    # Allow finalization even with some invalid fields (user confirmed)
    # But log the validation status
    if validation_result.invalid_count > 0:
        logger.warning(
            f"Finalizing extraction {extraction.id} with {validation_result.invalid_count} invalid fields"
        )
    
    try:
        # Finalize extraction
        finalized = await extraction_crud.finalize(db, extraction.id)
        
        # Also update document status to completed
        await document_crud.update_status(
            db,
            id=extraction.document_id,
            status=DocumentStatus.COMPLETED,
            form_type=extraction.form_type,
            language=extraction.language
        )
        
        await db.commit()
        
        logger.info(f"Extraction {extraction.id} finalized")
        
        return FinalizationResponse(
            extraction_id=extraction.id,
            is_finalized=True,
            finalized_at=finalized.finalized_at or datetime.utcnow(),
            message="Extraction finalized successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to finalize extraction {extraction.id}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to finalize extraction"
        )


@router.delete(
    "/{id}",
    summary="Delete extraction",
    description="""
    Delete an extraction and all its fields.
    
    Cannot delete finalized extractions.
    """,
    responses={
        200: {"description": "Extraction deleted"},
        404: {"description": "Extraction not found"},
        409: {"description": "Cannot delete finalized extraction"}
    }
)
async def delete_extraction(
    extraction: Extraction = Depends(get_extraction_or_404),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Delete an extraction.
    
    Reference: FastAPI Knowledge Base - Section 4
        - HTTPException for client errors
        - Consistent error response format
    """
    # Check not finalized
    check_not_finalized(extraction)
    
    try:
        await extraction_crud.delete(db, extraction.id)
        await db.commit()
        
        logger.info(f"Extraction {extraction.id} deleted")
        
        return {
            "id": extraction.id,
            "deleted": True,
            "message": "Extraction deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting extraction {extraction.id}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete extraction"
        )
