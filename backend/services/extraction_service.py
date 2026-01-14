"""
Extraction Service - LangGraph Workflow Orchestration
======================================================
Production-grade service using LangGraph Functional API to orchestrate
the complete document extraction pipeline: OCR → Gemini → Database.

Features:
- LangGraph @task and @entrypoint decorators for workflow management
- Built-in retry policies with exponential backoff per task
- Checkpointing for error recovery and resume capabilities
- Full async/sync support matching existing service patterns
- Integration with existing CRUD operations and processing logs

Usage:
    from services.extraction_service import extraction_service
    
    # Async extraction (recommended)
    result = await extraction_service.extract_document(
        document_id=doc_id,
        file_path="/path/to/file.pdf",
        file_type="pdf"
    )
    
    # Sync extraction
    result = extraction_service.extract_document_sync(
        document_id=doc_id,
        file_path="/path/to/file.pdf",
        file_type="pdf"
    )
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

# LangGraph Functional API imports
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy

# Internal service imports
from services.ocr_service import OCRService, DocumentOCRResult
from services.gemini_service import GeminiService, GeminiExtractionResult
from services.validation_service import ValidationService, BatchValidationResult

# Database imports
from database.crud import (
    document_crud,
    extraction_crud, 
    field_crud,
    processing_log_crud
)
from database.models import (
    DocumentStatus,
    ProcessingStep,
    LogStatus,
    FieldType
)
from database.connection import get_async_db

# Configuration
from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES - Pipeline State & Results
# =============================================================================

@dataclass
class ExtractionInput:
    """Input parameters for extraction workflow."""
    document_id: UUID
    file_path: str
    file_type: str
    custom_prompt: Optional[str] = None
    form_template: Optional[Dict[str, Any]] = None


@dataclass
class ExtractionResult:
    """Result from complete extraction pipeline."""
    extraction_id: Optional[UUID] = None
    document_id: Optional[UUID] = None
    
    # OCR Output
    raw_ocr_markdown: str = ""
    total_pages: int = 0
    
    # Gemini Output
    form_type: str = "Unknown"
    language: str = "en"
    fields: List[Dict[str, Any]] = field(default_factory=list)
    total_fields: int = 0
    
    # Processing Times (ms)
    ocr_time_ms: int = 0
    llm_time_ms: int = 0
    total_time_ms: int = 0
    
    # Status
    success: bool = True
    status: str = "completed"
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "extraction_id": str(self.extraction_id) if self.extraction_id else None,
            "document_id": str(self.document_id) if self.document_id else None,
            "raw_ocr_markdown": self.raw_ocr_markdown,
            "total_pages": self.total_pages,
            "form_type": self.form_type,
            "language": self.language,
            "fields": self.fields,
            "total_fields": self.total_fields,
            "ocr_time_ms": self.ocr_time_ms,
            "llm_time_ms": self.llm_time_ms,
            "total_time_ms": self.total_time_ms,
            "success": self.success,
            "status": self.status,
            "error": self.error
        }


@dataclass 
class OCRTaskOutput:
    """Output from OCR task."""
    markdown: str = ""
    html: str = ""
    total_pages: int = 0
    processing_time_ms: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class GeminiTaskOutput:
    """Output from Gemini extraction task."""
    fields: List[Dict[str, Any]] = field(default_factory=list)
    form_type: str = "Unknown"
    language: str = "en"
    processing_time_ms: int = 0
    token_count: int = 0
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# LANGGRAPH TASKS - Individual Pipeline Steps
# =============================================================================

# Retry policy for OCR task (may fail due to memory/GPU issues)
ocr_retry_policy = RetryPolicy(
    max_attempts=2,
    initial_interval=1.0,
    backoff_factor=2.0,
    retry_on=Exception  # Retry on any exception
)

# Retry policy for Gemini task (API errors, rate limits)
# Note: GeminiService has internal retry, this is for external failures
gemini_retry_policy = RetryPolicy(
    max_attempts=2,
    initial_interval=0.5,
    backoff_factor=2.0,
    retry_on=Exception
)


@task(retry=ocr_retry_policy)
async def run_ocr_task(file_path: str, file_type: str) -> OCRTaskOutput:
    """
    Execute OCR on a document using PaddleOCR-VL.
    
    This is an async task that wraps the OCR service.
    LangGraph handles execution properly.
    
    Args:
        file_path: Path to the document file
        file_type: File extension (pdf, png, jpg, jpeg)
        
    Returns:
        OCRTaskOutput with markdown text and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting OCR for: {file_path}")
        
        # Get OCR service instance (singleton)
        ocr_service = OCRService()
        
        # Process document - OCRService.process_document is async
        # We can await it directly since we're now an async function
        result: DocumentOCRResult = await ocr_service.process_document(file_path, file_type)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        if not result.success:
            logger.error(f"OCR failed: {result.error}")
            return OCRTaskOutput(
                success=False,
                error=result.error,
                processing_time_ms=processing_time
            )
        
        logger.info(f"OCR completed: {result.total_pages} pages in {processing_time}ms")
        
        return OCRTaskOutput(
            markdown=result.combined_markdown,
            html=result.combined_html,
            total_pages=result.total_pages,
            processing_time_ms=processing_time,
            success=True
        )
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        error_msg = f"OCR task error: {str(e)}"
        logger.exception(error_msg)
        return OCRTaskOutput(
            success=False,
            error=error_msg,
            processing_time_ms=processing_time
        )


@task(retry=gemini_retry_policy)
def run_gemini_extraction_task(
    ocr_text: str,
    custom_prompt: Optional[str] = None,
    form_template: Optional[Dict[str, Any]] = None
) -> GeminiTaskOutput:
    """
    Extract structured key-value pairs from OCR text using Gemini.
    
    Uses GeminiService with built-in retry logic for JSON parsing
    and API error handling.
    
    Args:
        ocr_text: Markdown text from OCR
        custom_prompt: Optional custom extraction prompt
        form_template: Optional form template for guided extraction
        
    Returns:
        GeminiTaskOutput with extracted fields
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting Gemini extraction ({len(ocr_text)} chars)")
        
        # Validate input
        if not ocr_text or not ocr_text.strip():
            return GeminiTaskOutput(
                success=False,
                error="Empty OCR text - nothing to extract",
                processing_time_ms=0
            )
        
        # Get Gemini service instance (singleton)
        gemini_service = GeminiService()
        
        # Extract synchronously (service handles retries internally)
        result: GeminiExtractionResult = gemini_service.extract_from_text(
            ocr_text=ocr_text,
            custom_prompt=custom_prompt,
            form_template=form_template,
            max_retries=2
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        if not result.success:
            logger.error(f"Gemini extraction failed: {result.error}")
            return GeminiTaskOutput(
                success=False,
                error=result.error,
                processing_time_ms=processing_time
            )
        
        logger.info(f"Gemini extraction completed: {len(result.fields)} fields in {processing_time}ms")
        
        return GeminiTaskOutput(
            fields=result.fields,
            form_type=result.form_type,
            language=result.language,
            processing_time_ms=processing_time,
            token_count=result.token_count,
            success=True
        )
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        error_msg = f"Gemini extraction error: {str(e)}"
        logger.exception(error_msg)
        return GeminiTaskOutput(
            success=False,
            error=error_msg,
            processing_time_ms=processing_time
        )


# =============================================================================
# DATABASE OPERATIONS (called within workflow, not as LangGraph tasks)
# =============================================================================

async def save_extraction_to_database(
    document_id: UUID,
    ocr_output: OCRTaskOutput,
    gemini_output: GeminiTaskOutput,
    total_time_ms: int
) -> Optional[UUID]:
    """
    Save extraction results to database.
    
    Creates versioned extraction record and bulk inserts fields.
    
    Args:
        document_id: Document UUID
        ocr_output: OCR task results
        gemini_output: Gemini task results
        total_time_ms: Total processing time
        
    Returns:
        Extraction UUID if successful, None otherwise
    """
    try:
        async with get_async_db() as db:
            # Create new extraction version
            extraction = await extraction_crud.create_new_version(
                db,
                document_id=document_id,
                raw_ocr_markdown=ocr_output.markdown,
                raw_ocr_html=ocr_output.html,
                form_type=gemini_output.form_type,
                language=gemini_output.language,
                ocr_time_ms=ocr_output.processing_time_ms,
                llm_time_ms=gemini_output.processing_time_ms,
                processing_time_ms=total_time_ms,
                status="completed",
                total_fields=len(gemini_output.fields),
                confidence_avg=_calculate_avg_confidence(gemini_output.fields)
            )
            
            # Prepare fields for bulk insert
            fields_data = []
            for i, field_data in enumerate(gemini_output.fields):
                fields_data.append({
                    "field_key": field_data.get("field_key", f"field_{i}"),
                    "field_value": field_data.get("field_value", ""),
                    "field_type": _map_field_type(field_data.get("field_type", "text")),
                    "confidence": field_data.get("confidence", 0.85),
                    "sort_order": i
                })
            
            # Bulk create fields
            if fields_data:
                await field_crud.bulk_create(db, extraction.id, fields_data)
            
            # Update document status
            await document_crud.update_status(
                db,
                id=document_id,
                status=DocumentStatus.COMPLETED,
                form_type=gemini_output.form_type,
                language=gemini_output.language
            )
            
            # Update extraction stats
            await extraction_crud.update_stats(db, extraction.id)
            
            await db.commit()
            
            logger.info(f"Saved extraction {extraction.id} with {len(fields_data)} fields")
            return extraction.id
            
    except Exception as e:
        logger.exception(f"Database save error: {e}")
        return None


async def log_processing_step(
    document_id: UUID,
    step: ProcessingStep,
    status: LogStatus,
    message: str = "",
    extraction_id: Optional[UUID] = None,
    duration_ms: Optional[int] = None,
    details: Optional[Dict] = None
):
    """Log a processing step to the database for audit trail."""
    try:
        async with get_async_db() as db:
            await processing_log_crud.log_step(
                db,
                document_id=document_id,
                step=step,
                status=status,
                message=message,
                extraction_id=extraction_id,
                duration_ms=duration_ms,
                details=details
            )
            await db.commit()
    except Exception as e:
        # Don't fail the workflow for logging errors
        logger.warning(f"Failed to log processing step: {e}")


async def mark_document_failed(document_id: UUID, error_message: str):
    """Mark document as failed in the database."""
    try:
        async with get_async_db() as db:
            await document_crud.update_status(
                db,
                id=document_id,
                status=DocumentStatus.FAILED
            )
            await db.commit()
    except Exception as e:
        logger.warning(f"Failed to mark document as failed: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _calculate_avg_confidence(fields: List[Dict[str, Any]]) -> float:
    """Calculate average confidence from field list."""
    if not fields:
        return 0.0
    confidences = [f.get("confidence", 0.85) for f in fields]
    return sum(confidences) / len(confidences)


def _map_field_type(type_str: str) -> FieldType:
    """Map string field type to FieldType enum."""
    type_mapping = {
        "text": FieldType.TEXT,
        "number": FieldType.NUMBER,
        "date": FieldType.DATE,
        "email": FieldType.EMAIL,
        "phone": FieldType.PHONE,
        "checkbox": FieldType.CHECKBOX,
        "table": FieldType.TABLE,
        "signature": FieldType.SIGNATURE,
        "address": FieldType.ADDRESS,
        "name": FieldType.NAME,
        "currency": FieldType.CURRENCY,
    }
    return type_mapping.get(type_str.lower(), FieldType.UNKNOWN)


# =============================================================================
# MAIN EXTRACTION WORKFLOW - LangGraph Entrypoint
# =============================================================================

# Checkpointer for workflow state persistence
checkpointer = InMemorySaver()


@entrypoint(checkpointer=checkpointer)
async def extraction_workflow(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main extraction workflow orchestrating OCR → Gemini → Database.
    
    This is the core LangGraph entrypoint that coordinates the full
    document extraction pipeline with proper error handling and
    checkpointing for recovery.
    
    Args:
        inputs: Dictionary with:
            - document_id: UUID of the document
            - file_path: Path to the document file
            - file_type: File extension (pdf, png, jpg, jpeg)
            - custom_prompt: Optional custom extraction prompt
            - form_template: Optional form template dict
            
    Returns:
        ExtractionResult as dictionary
    """
    start_time = time.time()
    
    # Parse inputs
    document_id = UUID(inputs["document_id"]) if isinstance(inputs["document_id"], str) else inputs["document_id"]
    file_path = inputs["file_path"]
    file_type = inputs["file_type"]
    custom_prompt = inputs.get("custom_prompt")
    form_template = inputs.get("form_template")
    
    logger.info(f"Starting extraction workflow for document {document_id}")
    
    # Initialize result
    result = ExtractionResult(document_id=document_id)
    
    try:
        # ===== STEP 1: OCR Processing =====
        await log_processing_step(
            document_id=document_id,
            step=ProcessingStep.OCR,
            status=LogStatus.STARTED,
            message="Starting OCR processing"
        )
        
        # Run OCR task (now async, await directly)
        ocr_output: OCRTaskOutput = await run_ocr_task(file_path, file_type)
        
        result.ocr_time_ms = ocr_output.processing_time_ms
        
        if not ocr_output.success:
            result.success = False
            result.status = "failed"
            result.error = f"OCR failed: {ocr_output.error}"
            
            await log_processing_step(
                document_id=document_id,
                step=ProcessingStep.OCR,
                status=LogStatus.FAILED,
                message=ocr_output.error or "OCR processing failed",
                duration_ms=ocr_output.processing_time_ms
            )
            await mark_document_failed(document_id, result.error)
            
            result.total_time_ms = int((time.time() - start_time) * 1000)
            return result.to_dict()
        
        result.raw_ocr_markdown = ocr_output.markdown
        result.total_pages = ocr_output.total_pages
        
        await log_processing_step(
            document_id=document_id,
            step=ProcessingStep.OCR,
            status=LogStatus.COMPLETED,
            message=f"OCR completed: {ocr_output.total_pages} pages",
            duration_ms=ocr_output.processing_time_ms
        )
        
        # ===== STEP 2: Gemini Extraction =====
        await log_processing_step(
            document_id=document_id,
            step=ProcessingStep.LLM_EXTRACTION,
            status=LogStatus.STARTED,
            message="Starting LLM extraction"
        )
        
        # Run Gemini extraction task (sync function, run in thread for non-blocking)
        gemini_output: GeminiTaskOutput = await asyncio.to_thread(
            run_gemini_extraction_task,
            ocr_output.markdown,
            custom_prompt,
            form_template
        )
        
        result.llm_time_ms = gemini_output.processing_time_ms
        
        if not gemini_output.success:
            result.success = False
            result.status = "failed"
            result.error = f"Gemini extraction failed: {gemini_output.error}"
            
            await log_processing_step(
                document_id=document_id,
                step=ProcessingStep.LLM_EXTRACTION,
                status=LogStatus.FAILED,
                message=gemini_output.error or "LLM extraction failed",
                duration_ms=gemini_output.processing_time_ms
            )
            await mark_document_failed(document_id, result.error)
            
            result.total_time_ms = int((time.time() - start_time) * 1000)
            return result.to_dict()
        
        result.form_type = gemini_output.form_type
        result.language = gemini_output.language
        result.fields = gemini_output.fields
        result.total_fields = len(gemini_output.fields)
        
        await log_processing_step(
            document_id=document_id,
            step=ProcessingStep.LLM_EXTRACTION,
            status=LogStatus.COMPLETED,
            message=f"Extracted {len(gemini_output.fields)} fields",
            duration_ms=gemini_output.processing_time_ms
        )
        
        # ===== STEP 3: Database Storage =====
        result.total_time_ms = int((time.time() - start_time) * 1000)
        
        extraction_id = await save_extraction_to_database(
            document_id=document_id,
            ocr_output=ocr_output,
            gemini_output=gemini_output,
            total_time_ms=result.total_time_ms
        )
        
        if extraction_id:
            result.extraction_id = extraction_id
            result.success = True
            result.status = "completed"
            
            await log_processing_step(
                document_id=document_id,
                step=ProcessingStep.FINALIZATION,
                status=LogStatus.COMPLETED,
                message="Extraction saved to database",
                extraction_id=extraction_id,
                duration_ms=result.total_time_ms
            )
            
            # ===== STEP 4: Automatic Validation =====
            try:
                await log_processing_step(
                    document_id=document_id,
                    step=ProcessingStep.VALIDATION,
                    status=LogStatus.STARTED,
                    message="Starting field validation",
                    extraction_id=extraction_id
                )
                
                validation_service = ValidationService()
                validation_result = await validation_service.validate_extraction(
                    extraction_id=extraction_id,
                    update_database=True
                )
                
                if validation_result.success:
                    logger.info(
                        f"Validation completed for {extraction_id}: "
                        f"{validation_result.valid_count}/{validation_result.total_fields} valid, "
                        f"{validation_result.needs_review_count} need review"
                    )
                else:
                    logger.warning(
                        f"Validation had issues for {extraction_id}: {validation_result.error}"
                    )
                    
            except Exception as val_error:
                # Validation failure shouldn't fail the whole extraction
                logger.warning(f"Validation step failed (non-critical): {val_error}")
                await log_processing_step(
                    document_id=document_id,
                    step=ProcessingStep.VALIDATION,
                    status=LogStatus.FAILED,
                    message=f"Validation failed: {str(val_error)}",
                    extraction_id=extraction_id
                )
            
            logger.info(
                f"Extraction workflow completed: {extraction_id} "
                f"({result.total_fields} fields in {result.total_time_ms}ms)"
            )
        else:
            result.success = False
            result.status = "failed"
            result.error = "Failed to save extraction to database"
            await mark_document_failed(document_id, result.error)
        
        return result.to_dict()
        
    except Exception as e:
        result.total_time_ms = int((time.time() - start_time) * 1000)
        result.success = False
        result.status = "failed"
        result.error = f"Workflow error: {str(e)}"
        
        logger.exception(f"Extraction workflow failed: {e}")
        
        await log_processing_step(
            document_id=document_id,
            step=ProcessingStep.FINALIZATION,
            status=LogStatus.FAILED,
            message=str(e),
            duration_ms=result.total_time_ms
        )
        await mark_document_failed(document_id, result.error)
        
        return result.to_dict()


# =============================================================================
# EXTRACTION SERVICE CLASS - Public API
# =============================================================================

class ExtractionService:
    """
    High-level extraction service providing clean public API.
    
    Wraps the LangGraph workflow for easy consumption by API endpoints.
    Follows singleton pattern consistent with other services.
    
    Usage:
        service = ExtractionService()
        result = await service.extract_document(doc_id, file_path, "pdf")
    """
    
    _instance: Optional["ExtractionService"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "ExtractionService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ExtractionService._initialized:
            return
        
        self._ocr_service = OCRService()
        self._gemini_service = GeminiService()
        
        ExtractionService._initialized = True
        logger.info("ExtractionService initialized")
    
    async def extract_document(
        self,
        document_id: UUID,
        file_path: Union[str, Path],
        file_type: str,
        custom_prompt: Optional[str] = None,
        form_template: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract structured data from a document asynchronously.
        
        Orchestrates the full pipeline: OCR → Gemini → Database.
        
        Args:
            document_id: UUID of the document in the database
            file_path: Path to the document file
            file_type: File extension (pdf, png, jpg, jpeg)
            custom_prompt: Optional custom extraction instructions
            form_template: Optional template for guided extraction
            thread_id: Optional thread ID for checkpointing
            
        Returns:
            ExtractionResult with extraction details and status
        """
        # Prepare inputs
        inputs = {
            "document_id": str(document_id),
            "file_path": str(file_path),
            "file_type": file_type.lower().lstrip("."),
            "custom_prompt": custom_prompt,
            "form_template": form_template
        }
        
        # Configure workflow
        config = {
            "configurable": {
                "thread_id": thread_id or str(document_id)
            }
        }
        
        try:
            # Mark document as processing
            async with get_async_db() as db:
                await document_crud.update_status(
                    db,
                    id=document_id,
                    status=DocumentStatus.PROCESSING
                )
                await db.commit()
            
            # Run workflow
            result_dict = await extraction_workflow.ainvoke(inputs, config=config)
            
            # Convert to ExtractionResult
            return ExtractionResult(
                extraction_id=UUID(result_dict["extraction_id"]) if result_dict.get("extraction_id") else None,
                document_id=UUID(result_dict["document_id"]) if result_dict.get("document_id") else None,
                raw_ocr_markdown=result_dict.get("raw_ocr_markdown", ""),
                total_pages=result_dict.get("total_pages", 0),
                form_type=result_dict.get("form_type", "Unknown"),
                language=result_dict.get("language", "en"),
                fields=result_dict.get("fields", []),
                total_fields=result_dict.get("total_fields", 0),
                ocr_time_ms=result_dict.get("ocr_time_ms", 0),
                llm_time_ms=result_dict.get("llm_time_ms", 0),
                total_time_ms=result_dict.get("total_time_ms", 0),
                success=result_dict.get("success", False),
                status=result_dict.get("status", "unknown"),
                error=result_dict.get("error")
            )
            
        except Exception as e:
            logger.exception(f"Extraction service error: {e}")
            return ExtractionResult(
                document_id=document_id,
                success=False,
                status="failed",
                error=str(e)
            )
    
    def extract_document_sync(
        self,
        document_id: UUID,
        file_path: Union[str, Path],
        file_type: str,
        custom_prompt: Optional[str] = None,
        form_template: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """
        Extract structured data from a document synchronously.
        
        Wrapper for extract_document that runs in event loop.
        
        Args:
            document_id: UUID of the document
            file_path: Path to the document file  
            file_type: File extension
            custom_prompt: Optional custom extraction instructions
            form_template: Optional template for guided extraction
            
        Returns:
            ExtractionResult with extraction details
        """
        # Use asyncio.run() which properly manages event loop lifecycle
        # This is the recommended approach for running async code from sync context
        return asyncio.run(
            self.extract_document(
                document_id=document_id,
                file_path=file_path,
                file_type=file_type,
                custom_prompt=custom_prompt,
                form_template=form_template
            )
        )
    
    async def get_extraction_status(
        self,
        extraction_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """
        Get the status of an extraction.
        
        Args:
            extraction_id: UUID of the extraction
            
        Returns:
            Dictionary with status info or None if not found
        """
        try:
            async with get_async_db() as db:
                extraction = await extraction_crud.get_with_fields(db, extraction_id)
                
                if not extraction:
                    return None
                
                return {
                    "id": str(extraction.id),
                    "document_id": str(extraction.document_id),
                    "version": extraction.version,
                    "status": extraction.status,
                    "form_type": extraction.form_type,
                    "language": extraction.language,
                    "total_fields": extraction.total_fields,
                    "confidence_avg": extraction.confidence_avg,
                    "processing_time_ms": extraction.processing_time_ms,
                    "is_finalized": extraction.is_finalized,
                    "created_at": extraction.created_at.isoformat() if extraction.created_at else None
                }
                
        except Exception as e:
            logger.error(f"Error getting extraction status: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status for health checks."""
        return {
            "service": "ExtractionService",
            "status": "ready",
            "ocr_service": self._ocr_service.get_status() if hasattr(self._ocr_service, 'get_status') else "unknown",
            "gemini_service": self._gemini_service.get_status() if hasattr(self._gemini_service, 'get_status') else "unknown"
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

extraction_service = ExtractionService()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ExtractionService",
    "extraction_service",
    "ExtractionResult",
    "ExtractionInput",
    "extraction_workflow"
]
