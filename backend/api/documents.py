"""
Documents API Router
====================
Handles document upload, retrieval, listing, and deletion.

Endpoints:
    POST   /documents/upload    - Upload a new document
    GET    /documents           - List all documents (paginated)
    GET    /documents/{id}      - Get document details
    DELETE /documents/{id}      - Soft delete a document

Integration:
    - Uses FileManager for file storage operations
    - Uses DocumentCRUD for database operations  
    - Uses Pydantic schemas from schemas/document.py

Reference: FastAPI Knowledge Base
    - File() and UploadFile for file uploads (requires python-multipart)
    - Path() for path parameters
    - Query() for query parameters
    - Depends() for dependency injection
    - response_model for response serialization
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Path
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List
from uuid import UUID
import logging

# Database
from database import (
    get_async_session,
    document_crud,
    Document,
    DocumentStatus,
    FileType,
)

# Schemas
from schemas.document import (
    DocumentResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
    DocumentDeleteResponse,
    DocumentUpdate,
    DocumentSearch,
)

# Utilities
from utils.file_manager import FileManager
from config import settings

# Logger
logger = logging.getLogger(__name__)

# =============================================================================
# ROUTER INSTANCE
# =============================================================================

router = APIRouter()

# File manager singleton
file_manager = FileManager()


# =============================================================================
# DEPENDENCIES
# =============================================================================

async def get_document_or_404(
    document_id: UUID = Path(..., description="Document ID"),
    db: AsyncSession = Depends(get_async_session)
) -> Document:
    """
    Dependency to get a document by ID or raise 404.
    
    Reference: FastAPI Knowledge Base - Section 2 (Dependency Injection)
        - Depends() for dependency injection
        - Path() for path parameters
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


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
    description="""
    Upload a document (PDF, PNG, JPG, JPEG) for processing.
    
    The file is validated for type and size, saved to storage,
    and a database record is created.
    
    **Supported formats:** PNG, JPG, JPEG, PDF  
    **Maximum size:** Configured via MAX_UPLOAD_SIZE_MB setting
    """,
    responses={
        201: {"description": "Document uploaded successfully"},
        400: {"description": "Invalid file type or size"},
        500: {"description": "Storage error"}
    }
)
async def upload_document(
    file: UploadFile = File(
        ...,
        description="Document file to upload (PDF, PNG, JPG, JPEG)"
    ),
    custom_filename: Optional[str] = Query(
        None,
        max_length=255,
        description="Custom filename (optional, will be sanitized)"
    ),
    auto_extract: bool = Query(
        False,
        description="Automatically start extraction after upload (not implemented yet)"
    ),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Upload a new document.
    
    Process:
    1. Validate file type and size
    2. Read file content
    3. Save to storage with unique filename
    4. Create database record
    5. Return upload response
    
    Reference: FastAPI Knowledge Base - Section 2
        - File() declares file uploads, requires python-multipart
        - UploadFile provides async methods for file handling
    """
    # Get filename
    original_filename = file.filename or "unknown_file"
    
    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file"
        )
    finally:
        await file.close()
    
    file_size = len(file_content)
    
    # Validate file
    is_valid, error_message = file_manager.validate_file(original_filename, file_size)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message
        )
    
    # Save file to storage
    try:
        file_path, stored_filename = file_manager.save_upload(
            file_content=file_content,
            original_filename=original_filename,
            custom_filename=custom_filename
        )
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save file to storage"
        )
    
    # Get relative path for database
    relative_path = file_manager.get_relative_path(file_path)
    
    # Determine file type
    extension = file_manager.get_extension(original_filename)
    try:
        file_type = FileType(extension)
    except ValueError:
        file_type = FileType.PDF  # Default fallback
    
    # Get MIME type
    mime_type = file_manager.get_mime_type(original_filename)
    
    # Get page count (placeholder - actual implementation would parse PDF)
    page_count = 1
    if file_type == FileType.PDF:
        # TODO: Use PyPDF2 to get actual page count
        page_count = 1
    
    # Create database record
    try:
        document = await document_crud.create(
            db,
            filename=stored_filename,
            original_filename=original_filename,
            file_path=str(relative_path),
            file_type=file_type,
            mime_type=mime_type,
            file_size_bytes=file_size,
            page_count=page_count,
            status=DocumentStatus.PENDING,
            custom_metadata={}
        )
        await db.commit()
        await db.refresh(document)
    except Exception as e:
        logger.error(f"Failed to create document record: {e}")
        # Clean up saved file
        file_manager.delete_file(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create document record"
        )
    
    logger.info(f"Document uploaded: {document.id} - {stored_filename}")
    
    return DocumentUploadResponse(
        id=document.id,
        filename=stored_filename,
        status=document.status,
        message="Document uploaded successfully",
        extraction_started=False  # TODO: Implement auto-extract
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents",
    description="""
    Get a paginated list of all non-deleted documents.
    
    Supports filtering by status and form type, and pagination.
    Results are ordered by creation date (newest first).
    """,
    responses={
        200: {"description": "List of documents"}
    }
)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[DocumentStatus] = Query(
        None,
        alias="status",
        description="Filter by processing status"
    ),
    form_type: Optional[str] = Query(
        None,
        max_length=100,
        description="Filter by form type"
    ),
    search: Optional[str] = Query(
        None,
        max_length=255,
        description="Search in filename and form type"
    ),
    db: AsyncSession = Depends(get_async_session)
):
    """
    List documents with pagination and optional filtering.
    
    Reference: FastAPI Knowledge Base - Section 2
        - Query() for query parameters with validation
        - response_model for automatic serialization
    """
    skip = (page - 1) * page_size
    
    # Get filtered documents
    if search:
        documents = await document_crud.search(
            db,
            query=search,
            status=status_filter,
            form_type=form_type,
            limit=page_size
        )
        # For search, we don't have total count optimization
        total = len(documents)
    elif status_filter:
        documents = await document_crud.get_by_status(db, status_filter, limit=page_size)
        total = len(documents)
    else:
        documents = await document_crud.get_active(db, skip=skip, limit=page_size)
        # Get total count for pagination
        total_result = await db.execute(
            select(func.count())
            .select_from(Document)
            .where(Document.is_deleted == False)
        )
        total = total_result.scalar() or 0
    
    # Calculate pagination
    pages = (total + page_size - 1) // page_size if total > 0 else 1
    
    # Convert to response items
    items = [
        DocumentListItem(
            id=doc.id,
            filename=doc.filename,
            original_filename=doc.original_filename,
            file_type=doc.file_type,
            file_size_bytes=doc.file_size_bytes,
            page_count=doc.page_count,
            status=doc.status,
            form_type=doc.form_type,
            created_at=doc.created_at
        )
        for doc in documents
    ]
    
    return DocumentListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
    description="""
    Get detailed information about a specific document.
    
    Includes file metadata, processing status, and extraction info.
    """,
    responses={
        200: {"description": "Document details"},
        404: {"description": "Document not found"}
    }
)
async def get_document(
    document: Document = Depends(get_document_or_404),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get a single document by ID.
    
    Reference: FastAPI Knowledge Base - Section 2
        - Depends() for reusable dependencies
        - Path() parameters are always required
    """
    # Get extraction info
    doc_with_extractions = await document_crud.get_with_extractions(db, document.id)
    
    current_extraction_id = None
    extraction_count = 0
    
    if doc_with_extractions and doc_with_extractions.extractions:
        extraction_count = len(doc_with_extractions.extractions)
        current_ext = doc_with_extractions.current_extraction
        if current_ext:
            current_extraction_id = current_ext.id
    
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        original_filename=document.original_filename,
        file_path=document.file_path,
        file_type=document.file_type,
        mime_type=document.mime_type,
        file_size_bytes=document.file_size_bytes,
        page_count=document.page_count,
        status=document.status,
        form_type=document.form_type,
        language=document.language,
        custom_metadata=document.custom_metadata or {},
        is_deleted=document.is_deleted,
        created_at=document.created_at,
        updated_at=document.updated_at,
        current_extraction_id=current_extraction_id,
        extraction_count=extraction_count
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    description="""
    Soft delete a document.
    
    The document is marked as deleted but data is retained.
    Associated extractions and files can be cleaned up later.
    """,
    responses={
        200: {"description": "Document deleted successfully"},
        404: {"description": "Document not found"}
    }
)
async def delete_document(
    document: Document = Depends(get_document_or_404),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Soft delete a document.
    
    Reference: FastAPI Knowledge Base - Section 4 (Error Management)
        - HTTPException for client-side errors
        - Consistent error response format
    """
    try:
        success = await document_crud.soft_delete(db, document.id)
        await db.commit()
        
        if success:
            logger.info(f"Document soft-deleted: {document.id}")
            return DocumentDeleteResponse(
                id=document.id,
                deleted=True,
                message="Document deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document.id}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the document"
        )
