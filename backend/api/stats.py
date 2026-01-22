"""
Stats API Router
================
Provides dashboard metrics and analytics for the OCR system.

Endpoints:
    GET    /stats/dashboard       - Main dashboard metrics
    GET    /stats/documents       - Document-specific stats
    GET    /stats/extractions     - Extraction-specific stats
    GET    /stats/processing      - Processing pipeline stats
    GET    /stats/system          - System health overview

Integration:
    - DocumentCRUD.get_stats() for document metrics
    - Direct SQL queries for aggregated stats
    - ProcessingLogCRUD for pipeline analytics

Reference: FastAPI Knowledge Base
    - response_model for automatic serialization
    - Query() for optional parameters
    - Direct database queries for aggregations
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, text
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
import logging

# Database
from database import (
    get_async_session,
    document_crud,
    extraction_crud,
    processing_log_crud,
    Document,
    Extraction,
    ExtractedField,
    ProcessingLog,
    DocumentStatus,
    LogStatus,
)

# Schemas
from schemas.document import DocumentStats

# Config
from config import settings

# Logger
logger = logging.getLogger(__name__)

# =============================================================================
# ROUTER INSTANCE
# =============================================================================

router = APIRouter()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_date_range(period: str) -> tuple[datetime, datetime]:
    """
    Get start and end datetime for a time period.
    
    Args:
        period: 'today', 'week', 'month', 'year'
        
    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    now = datetime.now(timezone.utc)
    end = now
    
    if period == "today":
        start = datetime(now.year, now.month, now.day)
    elif period == "week":
        start = now - timedelta(days=7)
    elif period == "month":
        start = now - timedelta(days=30)
    elif period == "year":
        start = now - timedelta(days=365)
    else:
        start = datetime(now.year, now.month, now.day)  # Default to today
    
    return start, end


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get(
    "/dashboard",
    summary="Get dashboard metrics",
    description="""
    Get comprehensive dashboard metrics for the OCR system.
    
    Returns:
    - Document counts (total, by status, by form type)
    - Extraction statistics (total, success rate, avg confidence)
    - Processing metrics (avg time, throughput)
    - Recent activity summary
    """,
    responses={
        200: {"description": "Dashboard metrics"}
    }
)
async def get_dashboard_stats(
    period: str = Query(
        "week",
        description="Time period: today, week, month, year"
    ),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get comprehensive dashboard metrics.
    
    Aggregates data from documents, extractions, and processing logs.
    
    Reference: FastAPI Knowledge Base - Section 3
        - Query() for parameter validation
        - Async database queries for performance
    """
    start_date, end_date = get_date_range(period)
    
    # ===== Document Stats =====
    doc_stats = await document_crud.get_stats(db)
    
    # Documents in period
    docs_in_period = await db.execute(
        select(func.count())
        .select_from(Document)
        .where(
            and_(
                Document.is_deleted == False,
                Document.created_at >= start_date,
                Document.created_at <= end_date
            )
        )
    )
    
    # Documents today
    today_start = datetime(datetime.now(timezone.utc).year, datetime.now(timezone.utc).month, datetime.now(timezone.utc).day, tzinfo=timezone.utc)
    docs_today = await db.execute(
        select(func.count())
        .select_from(Document)
        .where(
            and_(
                Document.is_deleted == False,
                Document.created_at >= today_start
            )
        )
    )
    
    # ===== Extraction Stats =====
    total_extractions = await db.execute(
        select(func.count())
        .select_from(Extraction)
    )
    
    completed_extractions = await db.execute(
        select(func.count())
        .select_from(Extraction)
        .where(Extraction.status == DocumentStatus.COMPLETED)
    )
    
    finalized_extractions = await db.execute(
        select(func.count())
        .select_from(Extraction)
        .where(Extraction.is_finalized == True)
    )
    
    # Average confidence
    avg_confidence = await db.execute(
        select(func.avg(Extraction.confidence_avg))
        .where(
            and_(
                Extraction.confidence_avg.isnot(None),
                Extraction.status == DocumentStatus.COMPLETED
            )
        )
    )
    
    # ===== Processing Time Stats =====
    avg_processing_time = await db.execute(
        select(func.avg(Extraction.processing_time_ms))
        .where(Extraction.processing_time_ms.isnot(None))
    )
    
    avg_ocr_time = await db.execute(
        select(func.avg(Extraction.ocr_time_ms))
        .where(Extraction.ocr_time_ms.isnot(None))
    )
    
    avg_llm_time = await db.execute(
        select(func.avg(Extraction.llm_time_ms))
        .where(Extraction.llm_time_ms.isnot(None))
    )
    
    # ===== Field Stats =====
    total_fields = await db.execute(
        select(func.count())
        .select_from(ExtractedField)
    )
    
    edited_fields = await db.execute(
        select(func.count())
        .select_from(ExtractedField)
        .where(ExtractedField.is_edited == True)
    )
    
    # ===== Build Response =====
    total_ext = total_extractions.scalar() or 0
    completed_ext = completed_extractions.scalar() or 0
    
    return {
        "period": period,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        
        # Document metrics
        "documents": {
            "total": doc_stats.get("total", 0),
            "in_period": docs_in_period.scalar() or 0,
            "today": docs_today.scalar() or 0,
            "by_status": doc_stats.get("by_status", {}),
            "by_form_type": doc_stats.get("by_form_type", {})
        },
        
        # Extraction metrics
        "extractions": {
            "total": total_ext,
            "completed": completed_ext,
            "finalized": finalized_extractions.scalar() or 0,
            "success_rate": round((completed_ext / total_ext * 100), 1) if total_ext > 0 else 0,
            "avg_confidence": round((avg_confidence.scalar() or 0) * 100, 1)
        },
        
        # Processing metrics
        "processing": {
            "avg_total_time_ms": round(avg_processing_time.scalar() or 0, 0),
            "avg_ocr_time_ms": round(avg_ocr_time.scalar() or 0, 0),
            "avg_llm_time_ms": round(avg_llm_time.scalar() or 0, 0)
        },
        
        # Field metrics
        "fields": {
            "total_extracted": total_fields.scalar() or 0,
            "total_edited": edited_fields.scalar() or 0,
            "edit_rate": round(
                (edited_fields.scalar() or 0) / (total_fields.scalar() or 1) * 100, 1
            )
        }
    }


@router.get(
    "/documents",
    response_model=DocumentStats,
    summary="Get document statistics",
    description="""
    Get detailed document statistics.
    
    Includes counts by status, form type, and time-based metrics.
    """,
    responses={
        200: {"description": "Document statistics"}
    }
)
async def get_document_stats(
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get document-specific statistics.
    
    Uses DocumentCRUD.get_stats() for core metrics.
    """
    stats = await document_crud.get_stats(db)
    
    # Additional calculations
    now_utc = datetime.now(timezone.utc)
    today_start = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
    week_start = now_utc - timedelta(days=7)
    
    docs_today = await db.execute(
        select(func.count())
        .select_from(Document)
        .where(
            and_(
                Document.is_deleted == False,
                Document.created_at >= today_start
            )
        )
    )
    
    docs_week = await db.execute(
        select(func.count())
        .select_from(Document)
        .where(
            and_(
                Document.is_deleted == False,
                Document.created_at >= week_start
            )
        )
    )
    
    total_pages = await db.execute(
        select(func.sum(Document.page_count))
        .where(Document.is_deleted == False)
    )
    
    avg_processing = await db.execute(
        select(func.avg(Extraction.processing_time_ms))
        .where(Extraction.processing_time_ms.isnot(None))
    )
    
    return DocumentStats(
        total_documents=stats.get("total", 0),
        documents_today=docs_today.scalar() or 0,
        documents_this_week=docs_week.scalar() or 0,
        by_status=stats.get("by_status", {}),
        by_form_type=stats.get("by_form_type", {}),
        total_pages_processed=total_pages.scalar() or 0,
        avg_processing_time_ms=avg_processing.scalar()
    )


@router.get(
    "/extractions",
    summary="Get extraction statistics",
    description="""
    Get detailed extraction statistics.
    
    Includes completion rates, confidence distributions, and more.
    """,
    responses={
        200: {"description": "Extraction statistics"}
    }
)
async def get_extraction_stats(
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get extraction-specific statistics.
    """
    # Total counts
    total = await db.execute(
        select(func.count()).select_from(Extraction)
    )
    
    # By status
    by_status = await db.execute(
        select(Extraction.status, func.count())
        .group_by(Extraction.status)
    )
    
    # Finalized
    finalized = await db.execute(
        select(func.count())
        .select_from(Extraction)
        .where(Extraction.is_finalized == True)
    )
    
    # Confidence distribution
    high_confidence = await db.execute(
        select(func.count())
        .select_from(Extraction)
        .where(
            and_(
                Extraction.confidence_avg.isnot(None),
                Extraction.confidence_avg >= 0.85
            )
        )
    )
    
    medium_confidence = await db.execute(
        select(func.count())
        .select_from(Extraction)
        .where(
            and_(
                Extraction.confidence_avg.isnot(None),
                Extraction.confidence_avg >= 0.60,
                Extraction.confidence_avg < 0.85
            )
        )
    )
    
    low_confidence = await db.execute(
        select(func.count())
        .select_from(Extraction)
        .where(
            and_(
                Extraction.confidence_avg.isnot(None),
                Extraction.confidence_avg < 0.60
            )
        )
    )
    
    # Average stats
    avg_fields = await db.execute(
        select(func.avg(Extraction.total_fields))
        .where(Extraction.total_fields.isnot(None))
    )
    
    avg_confidence = await db.execute(
        select(func.avg(Extraction.confidence_avg))
        .where(Extraction.confidence_avg.isnot(None))
    )
    
    return {
        "total": total.scalar() or 0,
        "finalized": finalized.scalar() or 0,
        "by_status": {
            row[0].value if row[0] else "unknown": row[1] 
            for row in by_status
        },
        "confidence_distribution": {
            "high": high_confidence.scalar() or 0,
            "medium": medium_confidence.scalar() or 0,
            "low": low_confidence.scalar() or 0
        },
        "averages": {
            "fields_per_extraction": round(avg_fields.scalar() or 0, 1),
            "confidence": round((avg_confidence.scalar() or 0) * 100, 1)
        }
    }


@router.get(
    "/processing",
    summary="Get processing pipeline statistics",
    description="""
    Get processing pipeline performance metrics.
    
    Includes timing breakdowns, success/failure rates, and bottlenecks.
    """,
    responses={
        200: {"description": "Processing statistics"}
    }
)
async def get_processing_stats(
    limit: int = Query(50, ge=1, le=100, description="Number of recent logs"),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get processing pipeline statistics.
    """
    # Recent failed steps
    failed_logs = await processing_log_crud.get_failed_steps(db, limit=limit)
    
    # Processing time stats
    avg_times = await db.execute(
        select(
            func.avg(Extraction.processing_time_ms).label("total"),
            func.avg(Extraction.ocr_time_ms).label("ocr"),
            func.avg(Extraction.llm_time_ms).label("llm"),
            func.min(Extraction.processing_time_ms).label("min_total"),
            func.max(Extraction.processing_time_ms).label("max_total")
        )
        .where(Extraction.processing_time_ms.isnot(None))
    )
    time_row = avg_times.first()
    
    # Success rate by step (from processing logs)
    step_stats = await db.execute(
        select(
            ProcessingLog.step,
            ProcessingLog.status,
            func.count()
        )
        .group_by(ProcessingLog.step, ProcessingLog.status)
    )
    
    # Organize step stats
    steps_summary = {}
    for row in step_stats:
        step_name = row[0].value if row[0] else "unknown"
        status_name = row[1].value if row[1] else "unknown"
        
        if step_name not in steps_summary:
            steps_summary[step_name] = {}
        steps_summary[step_name][status_name] = row[2]
    
    return {
        "timing": {
            "avg_total_ms": round(time_row.total or 0, 0) if time_row else 0,
            "avg_ocr_ms": round(time_row.ocr or 0, 0) if time_row else 0,
            "avg_llm_ms": round(time_row.llm or 0, 0) if time_row else 0,
            "min_total_ms": time_row.min_total if time_row else 0,
            "max_total_ms": time_row.max_total if time_row else 0
        },
        "steps": steps_summary,
        "recent_failures": [
            {
                "id": str(log.id),
                "document_id": str(log.document_id),
                "step": log.step.value if log.step else "unknown",
                "message": log.message,
                "created_at": log.created_at.isoformat() if log.created_at else None
            }
            for log in failed_logs[:10]  # Top 10 recent failures
        ],
        "failure_count": len(failed_logs)
    }


@router.get(
    "/system",
    summary="Get system overview",
    description="""
    Get system health and configuration overview.
    
    Combines all key metrics into a single summary.
    """,
    responses={
        200: {"description": "System overview"}
    }
)
async def get_system_overview(
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get system overview for health monitoring.
    """
    # Quick counts
    doc_count = await db.execute(
        select(func.count())
        .select_from(Document)
        .where(Document.is_deleted == False)
    )
    
    extraction_count = await db.execute(
        select(func.count()).select_from(Extraction)
    )
    
    pending_count = await db.execute(
        select(func.count())
        .select_from(Document)
        .where(
            and_(
                Document.is_deleted == False,
                Document.status == DocumentStatus.PENDING
            )
        )
    )
    
    processing_count = await db.execute(
        select(func.count())
        .select_from(Document)
        .where(
            and_(
                Document.is_deleted == False,
                Document.status == DocumentStatus.PROCESSING
            )
        )
    )
    
    failed_count = await db.execute(
        select(func.count())
        .select_from(Document)
        .where(
            and_(
                Document.is_deleted == False,
                Document.status == DocumentStatus.FAILED
            )
        )
    )
    
    # Storage check
    storage_ok = (
        settings.UPLOAD_PATH.exists() and
        settings.PROCESSED_PATH.exists() and
        settings.EXPORT_PATH.exists()
    )
    
    # Gemini config check
    gemini_configured = bool(settings.GEMINI_API_KEY)
    
    return {
        "status": "healthy" if storage_ok and gemini_configured else "degraded",
        "app": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "version": "1.0.0",
        
        "counts": {
            "documents": doc_count.scalar() or 0,
            "extractions": extraction_count.scalar() or 0,
            "pending": pending_count.scalar() or 0,
            "processing": processing_count.scalar() or 0,
            "failed": failed_count.scalar() or 0
        },
        
        "services": {
            "database": "connected",
            "storage": "ok" if storage_ok else "error",
            "gemini": "configured" if gemini_configured else "not_configured"
        },
        
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
