"""
Database CRUD Operations
========================
Comprehensive Create, Read, Update, Delete operations for all models.
Both synchronous and asynchronous versions provided.

Usage:
    from database.crud import document_crud, extraction_crud
    
    # Async (FastAPI)
    doc = await document_crud.create(db, data)
    
    # Sync
    with get_db() as db:
        doc = document_crud.create_sync(db, data)
"""

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import Session, selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any, TypeVar, Generic, Type
from datetime import datetime, timezone
from uuid import UUID
import logging

from database.models import (
    Document, Extraction, ExtractedField, FieldEdit, 
    FormTemplate, ProcessingLog, DocumentStatus, FieldType,
    EditType, ProcessingStep, LogStatus
)

logger = logging.getLogger(__name__)

# Type variable for generic CRUD
ModelType = TypeVar("ModelType")


# =============================================================================
# BASE CRUD CLASS
# =============================================================================

class BaseCRUD(Generic[ModelType]):
    """
    Base class with common CRUD operations.
    Inherit for model-specific operations.
    """
    
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    # ===== ASYNC OPERATIONS =====
    
    async def get(self, db: AsyncSession, id: UUID) -> Optional[ModelType]:
        """Get single record by ID"""
        result = await db.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_multi(
        self, 
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100,
        order_by: str = "created_at",
        descending: bool = True
    ) -> List[ModelType]:
        """Get multiple records with pagination"""
        order_col = getattr(self.model, order_by, self.model.created_at)
        order = order_col.desc() if descending else order_col.asc()
        
        result = await db.execute(
            select(self.model).order_by(order).offset(skip).limit(limit)
        )
        return list(result.scalars().all())
    
    async def create(self, db: AsyncSession, **data) -> ModelType:
        """Create new record"""
        obj = self.model(**data)
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj
    
    async def update(
        self, 
        db: AsyncSession, 
        id: UUID, 
        **data
    ) -> Optional[ModelType]:
        """Update record by ID"""
        obj = await self.get(db, id)
        if obj:
            for key, value in data.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            await db.flush()
            await db.refresh(obj)
        return obj
    
    async def delete(self, db: AsyncSession, id: UUID) -> bool:
        """Hard delete record by ID"""
        result = await db.execute(
            delete(self.model).where(self.model.id == id)
        )
        return result.rowcount > 0
    
    async def count(self, db: AsyncSession) -> int:
        """Count total records"""
        result = await db.execute(
            select(func.count()).select_from(self.model)
        )
        return result.scalar() or 0
    
    # ===== SYNC OPERATIONS =====
    
    def get_sync(self, db: Session, id: UUID) -> Optional[ModelType]:
        """Sync: Get single record by ID"""
        return db.query(self.model).filter(self.model.id == id).first()
    
    def get_multi_sync(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[ModelType]:
        """Sync: Get multiple records"""
        return db.query(self.model).offset(skip).limit(limit).all()
    
    def create_sync(self, db: Session, **data) -> ModelType:
        """Sync: Create new record"""
        obj = self.model(**data)
        db.add(obj)
        db.flush()
        db.refresh(obj)
        return obj
    
    def update_sync(self, db: Session, id: UUID, **data) -> Optional[ModelType]:
        """Sync: Update record"""
        obj = self.get_sync(db, id)
        if obj:
            for key, value in data.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            db.flush()
            db.refresh(obj)
        return obj
    
    def delete_sync(self, db: Session, id: UUID) -> bool:
        """Sync: Hard delete record"""
        obj = self.get_sync(db, id)
        if obj:
            db.delete(obj)
            return True
        return False


# =============================================================================
# DOCUMENT CRUD
# =============================================================================

class DocumentCRUD(BaseCRUD[Document]):
    """CRUD operations for Document model"""
    
    def __init__(self):
        super().__init__(Document)
    
    async def get_with_extractions(
        self, 
        db: AsyncSession, 
        id: UUID
    ) -> Optional[Document]:
        """Get document with all extractions loaded"""
        result = await db.execute(
            select(Document)
            .options(selectinload(Document.extractions))
            .where(Document.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_active(
        self, 
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Document]:
        """Get non-deleted documents"""
        result = await db.execute(
            select(Document)
            .where(Document.is_deleted == False)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_by_status(
        self, 
        db: AsyncSession, 
        status: DocumentStatus,
        limit: int = 100
    ) -> List[Document]:
        """Get documents by processing status"""
        result = await db.execute(
            select(Document)
            .where(
                and_(
                    Document.status == status,
                    Document.is_deleted == False
                )
            )
            .order_by(Document.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def soft_delete(self, db: AsyncSession, id: UUID) -> bool:
        """Soft delete document"""
        result = await db.execute(
            update(Document)
            .where(Document.id == id)
            .values(is_deleted=True, deleted_at=datetime.now(timezone.utc))
        )
        return result.rowcount > 0
    
    async def update_status(
        self, 
        db: AsyncSession,
        id: UUID, 
        status: DocumentStatus,
        form_type: Optional[str] = None,
        language: Optional[str] = None
    ) -> Optional[Document]:
        """Update document processing status"""
        update_data = {"status": status}
        if form_type:
            update_data["form_type"] = form_type
        if language:
            update_data["language"] = language
        
        return await self.update(db, id, **update_data)
    
    async def search(
        self, 
        db: AsyncSession,
        query: str,
        status: Optional[DocumentStatus] = None,
        form_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Document]:
        """Search documents by filename or form type"""
        conditions = [
            Document.is_deleted == False,
            or_(
                Document.filename.ilike(f"%{query}%"),
                Document.original_filename.ilike(f"%{query}%"),
                Document.form_type.ilike(f"%{query}%")
            )
        ]
        
        if status:
            conditions.append(Document.status == status)
        if form_type:
            conditions.append(Document.form_type == form_type)
        
        result = await db.execute(
            select(Document)
            .where(and_(*conditions))
            .order_by(Document.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
    
    async def get_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """Get document statistics"""
        # Total documents
        total = await db.execute(
            select(func.count()).select_from(Document)
            .where(Document.is_deleted == False)
        )
        
        # By status
        status_counts = await db.execute(
            select(Document.status, func.count())
            .where(Document.is_deleted == False)
            .group_by(Document.status)
        )
        
        # By form type (top 10)
        form_type_counts = await db.execute(
            select(Document.form_type, func.count())
            .where(
                and_(
                    Document.is_deleted == False,
                    Document.form_type.isnot(None)
                )
            )
            .group_by(Document.form_type)
            .order_by(func.count().desc())
            .limit(10)
        )
        
        return {
            "total": total.scalar() or 0,
            "by_status": {row[0].value: row[1] for row in status_counts},
            "by_form_type": {row[0]: row[1] for row in form_type_counts}
        }


# =============================================================================
# EXTRACTION CRUD
# =============================================================================

class ExtractionCRUD(BaseCRUD[Extraction]):
    """CRUD operations for Extraction model"""
    
    def __init__(self):
        super().__init__(Extraction)
    
    async def get_with_fields(
        self, 
        db: AsyncSession, 
        id: UUID
    ) -> Optional[Extraction]:
        """Get extraction with all fields loaded"""
        result = await db.execute(
            select(Extraction)
            .options(selectinload(Extraction.fields))
            .where(Extraction.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_current_for_document(
        self, 
        db: AsyncSession, 
        document_id: UUID
    ) -> Optional[Extraction]:
        """Get the current extraction for a document"""
        result = await db.execute(
            select(Extraction)
            .options(selectinload(Extraction.fields))
            .where(
                and_(
                    Extraction.document_id == document_id,
                    Extraction.is_current == True
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def get_all_versions(
        self, 
        db: AsyncSession, 
        document_id: UUID
    ) -> List[Extraction]:
        """Get all extraction versions for a document"""
        result = await db.execute(
            select(Extraction)
            .where(Extraction.document_id == document_id)
            .order_by(Extraction.version.desc())
        )
        return list(result.scalars().all())
    
    async def create_new_version(
        self, 
        db: AsyncSession, 
        document_id: UUID,
        **data
    ) -> Extraction:
        """Create new extraction version, marking previous as not current"""
        # Get current max version
        result = await db.execute(
            select(func.max(Extraction.version))
            .where(Extraction.document_id == document_id)
        )
        max_version = result.scalar() or 0
        
        # Mark all previous as not current
        await db.execute(
            update(Extraction)
            .where(Extraction.document_id == document_id)
            .values(is_current=False)
        )
        
        # Create new version
        return await self.create(
            db,
            document_id=document_id,
            version=max_version + 1,
            is_current=True,
            **data
        )
    
    async def finalize(
        self, 
        db: AsyncSession, 
        id: UUID
    ) -> Optional[Extraction]:
        """Mark extraction as finalized"""
        return await self.update(
            db, id, 
            is_finalized=True, 
            finalized_at=datetime.now(timezone.utc)
        )
    
    async def update_stats(
        self, 
        db: AsyncSession, 
        id: UUID
    ) -> Optional[Extraction]:
        """Recalculate and update extraction statistics"""
        extraction = await self.get_with_fields(db, id)
        if not extraction:
            return None
        
        fields = extraction.fields
        total = len(fields)
        edited = sum(1 for f in fields if f.is_edited)
        avg_confidence = (
            sum(f.confidence or 0 for f in fields) / total 
            if total > 0 else 0
        )
        
        return await self.update(
            db, id,
            total_fields=total,
            edited_fields_count=edited,
            confidence_avg=avg_confidence
        )
    
    async def get_finalized(
        self, 
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[Extraction]:
        """Get all finalized extractions"""
        result = await db.execute(
            select(Extraction)
            .options(joinedload(Extraction.document))
            .where(Extraction.is_finalized == True)
            .order_by(Extraction.finalized_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


# =============================================================================
# EXTRACTED FIELD CRUD
# =============================================================================

class ExtractedFieldCRUD(BaseCRUD[ExtractedField]):
    """CRUD operations for ExtractedField model"""
    
    def __init__(self):
        super().__init__(ExtractedField)
    
    async def get_by_extraction(
        self, 
        db: AsyncSession, 
        extraction_id: UUID
    ) -> List[ExtractedField]:
        """Get all fields for an extraction"""
        result = await db.execute(
            select(ExtractedField)
            .where(ExtractedField.extraction_id == extraction_id)
            .order_by(ExtractedField.sort_order)
        )
        return list(result.scalars().all())
    
    async def bulk_create(
        self, 
        db: AsyncSession, 
        extraction_id: UUID,
        fields_data: List[Dict[str, Any]]
    ) -> List[ExtractedField]:
        """Create multiple fields at once"""
        fields = []
        for i, data in enumerate(fields_data):
            field = ExtractedField(
                extraction_id=extraction_id,
                sort_order=i,
                **data
            )
            db.add(field)
            fields.append(field)
        
        await db.flush()
        for field in fields:
            await db.refresh(field)
        
        return fields
    
    async def update_value(
        self, 
        db: AsyncSession, 
        id: UUID,
        new_value: str,
        edit_type: EditType = EditType.MANUAL,
        edit_source: str = "user"
    ) -> Optional[ExtractedField]:
        """Update field value with edit tracking"""
        field = await self.get(db, id)
        if not field:
            return None
        
        old_value = field.field_value
        
        # Store original if first edit
        if not field.is_edited:
            field.original_value = old_value
        
        # Update field
        field.field_value = new_value
        field.is_edited = True
        field.updated_at = datetime.now(timezone.utc)
        
        # Create edit record
        edit = FieldEdit(
            field_id=field.id,
            extraction_id=field.extraction_id,
            old_value=old_value,
            new_value=new_value,
            edit_type=edit_type,
            edit_source=edit_source
        )
        db.add(edit)
        
        await db.flush()
        await db.refresh(field)
        
        return field
    
    async def bulk_update(
        self, 
        db: AsyncSession,
        updates: List[Dict[str, Any]]
    ) -> List[ExtractedField]:
        """
        Bulk update multiple fields.
        Each dict should have 'id' and fields to update.
        """
        updated_fields = []
        for update_data in updates:
            field_id = update_data.pop("id")
            new_value = update_data.get("field_value")
            
            if new_value is not None:
                field = await self.update_value(db, field_id, new_value)
            else:
                field = await self.update(db, field_id, **update_data)
            
            if field:
                updated_fields.append(field)
        
        return updated_fields
    
    async def get_edited_fields(
        self, 
        db: AsyncSession, 
        extraction_id: UUID
    ) -> List[ExtractedField]:
        """Get only edited fields for an extraction"""
        result = await db.execute(
            select(ExtractedField)
            .where(
                and_(
                    ExtractedField.extraction_id == extraction_id,
                    ExtractedField.is_edited == True
                )
            )
            .order_by(ExtractedField.sort_order)
        )
        return list(result.scalars().all())
    
    async def get_low_confidence_fields(
        self, 
        db: AsyncSession, 
        extraction_id: UUID,
        threshold: float = 0.6
    ) -> List[ExtractedField]:
        """Get fields with confidence below threshold"""
        result = await db.execute(
            select(ExtractedField)
            .where(
                and_(
                    ExtractedField.extraction_id == extraction_id,
                    ExtractedField.confidence < threshold
                )
            )
            .order_by(ExtractedField.confidence)
        )
        return list(result.scalars().all())


# =============================================================================
# FIELD EDIT CRUD
# =============================================================================

class FieldEditCRUD(BaseCRUD[FieldEdit]):
    """CRUD operations for FieldEdit model (audit trail)"""
    
    def __init__(self):
        super().__init__(FieldEdit)
    
    async def get_by_field(
        self, 
        db: AsyncSession, 
        field_id: UUID
    ) -> List[FieldEdit]:
        """Get edit history for a field"""
        result = await db.execute(
            select(FieldEdit)
            .where(FieldEdit.field_id == field_id)
            .order_by(FieldEdit.created_at.desc())
        )
        return list(result.scalars().all())
    
    async def get_by_extraction(
        self, 
        db: AsyncSession, 
        extraction_id: UUID
    ) -> List[FieldEdit]:
        """Get all edits for an extraction"""
        result = await db.execute(
            select(FieldEdit)
            .where(FieldEdit.extraction_id == extraction_id)
            .order_by(FieldEdit.created_at.desc())
        )
        return list(result.scalars().all())
    
    async def get_recent(
        self, 
        db: AsyncSession, 
        limit: int = 50
    ) -> List[FieldEdit]:
        """Get recent edits across all documents"""
        result = await db.execute(
            select(FieldEdit)
            .options(joinedload(FieldEdit.field))
            .order_by(FieldEdit.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# =============================================================================
# FORM TEMPLATE CRUD
# =============================================================================

class FormTemplateCRUD(BaseCRUD[FormTemplate]):
    """CRUD operations for FormTemplate model"""
    
    def __init__(self):
        super().__init__(FormTemplate)
    
    async def get_by_name(
        self, 
        db: AsyncSession, 
        name: str
    ) -> Optional[FormTemplate]:
        """Get template by name"""
        result = await db.execute(
            select(FormTemplate)
            .where(FormTemplate.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_active(
        self, 
        db: AsyncSession
    ) -> List[FormTemplate]:
        """Get all active templates"""
        result = await db.execute(
            select(FormTemplate)
            .where(FormTemplate.is_active == True)
            .order_by(FormTemplate.usage_count.desc())
        )
        return list(result.scalars().all())
    
    async def increment_usage(
        self, 
        db: AsyncSession, 
        id: UUID
    ) -> Optional[FormTemplate]:
        """Increment template usage count"""
        result = await db.execute(
            update(FormTemplate)
            .where(FormTemplate.id == id)
            .values(usage_count=FormTemplate.usage_count + 1)
            .returning(FormTemplate)
        )
        return result.scalar_one_or_none()


# =============================================================================
# PROCESSING LOG CRUD
# =============================================================================

class ProcessingLogCRUD(BaseCRUD[ProcessingLog]):
    """CRUD operations for ProcessingLog model"""
    
    def __init__(self):
        super().__init__(ProcessingLog)
    
    async def log_step(
        self,
        db: AsyncSession,
        document_id: UUID,
        step: ProcessingStep,
        status: LogStatus,
        message: str = "",
        extraction_id: Optional[UUID] = None,
        details: Optional[Dict] = None,
        duration_ms: Optional[int] = None
    ) -> ProcessingLog:
        """Create a processing log entry"""
        return await self.create(
            db,
            document_id=document_id,
            extraction_id=extraction_id,
            step=step,
            status=status,
            message=message,
            details=details or {},
            duration_ms=duration_ms
        )
    
    async def get_by_document(
        self, 
        db: AsyncSession, 
        document_id: UUID
    ) -> List[ProcessingLog]:
        """Get all logs for a document"""
        result = await db.execute(
            select(ProcessingLog)
            .where(ProcessingLog.document_id == document_id)
            .order_by(ProcessingLog.created_at)
        )
        return list(result.scalars().all())
    
    async def get_failed_steps(
        self, 
        db: AsyncSession, 
        limit: int = 50
    ) -> List[ProcessingLog]:
        """Get recent failed processing steps"""
        result = await db.execute(
            select(ProcessingLog)
            .where(ProcessingLog.status == LogStatus.FAILED)
            .order_by(ProcessingLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# =============================================================================
# CRUD INSTANCES (Ready to use)
# =============================================================================

document_crud = DocumentCRUD()
extraction_crud = ExtractionCRUD()
field_crud = ExtractedFieldCRUD()
field_edit_crud = FieldEditCRUD()
template_crud = FormTemplateCRUD()
processing_log_crud = ProcessingLogCRUD()
