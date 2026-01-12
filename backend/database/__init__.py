"""
Database Package
================
Provides database connection, ORM models, and CRUD operations.

Quick Start:
    from database import (
        # Connection
        get_db, get_async_session, create_all_tables,
        
        # Models
        Document, Extraction, ExtractedField, FieldEdit,
        FormTemplate, ProcessingLog,
        
        # Enums
        DocumentStatus, FieldType, EditType,
        
        # CRUD
        document_crud, extraction_crud, field_crud
    )

FastAPI Dependency:
    from database import get_async_session
    
    @app.get("/documents")
    async def list_docs(db: AsyncSession = Depends(get_async_session)):
        return await document_crud.get_active(db)
"""

# Connection & Session Management
from database.connection import (
    Base,
    get_db,
    get_sync_session,
    get_async_db,
    get_async_session,
    create_all_tables,
    async_create_all_tables,
    drop_all_tables,
    check_database_connection,
    dispose_engines,
    async_dispose_engines,
    sync_engine,
    async_engine,
)

# ORM Models
from database.models import (
    Document,
    Extraction,
    ExtractedField,
    FieldEdit,
    FormTemplate,
    ProcessingLog,
)

# Enums
from database.models import (
    DocumentStatus,
    FileType,
    FieldType,
    EditType,
    ProcessingStep,
    LogStatus,
)

# CRUD Operations
from database.crud import (
    document_crud,
    extraction_crud,
    field_crud,
    field_edit_crud,
    template_crud,
    processing_log_crud,
)

__all__ = [
    # Connection
    "Base",
    "get_db",
    "get_sync_session",
    "get_async_db",
    "get_async_session",
    "create_all_tables",
    "async_create_all_tables",
    "drop_all_tables",
    "check_database_connection",
    "dispose_engines",
    "async_dispose_engines",
    "sync_engine",
    "async_engine",
    
    # Models
    "Document",
    "Extraction",
    "ExtractedField",
    "FieldEdit",
    "FormTemplate",
    "ProcessingLog",
    
    # Enums
    "DocumentStatus",
    "FileType",
    "FieldType",
    "EditType",
    "ProcessingStep",
    "LogStatus",
    
    # CRUD
    "document_crud",
    "extraction_crud",
    "field_crud",
    "field_edit_crud",
    "template_crud",
    "processing_log_crud",
]
