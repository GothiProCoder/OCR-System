"""
FastAPI Application Entry Point
===============================
Main application setup with CORS, routers, and lifecycle management.

Usage:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import time

from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# import logging

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)


# =============================================================================
# APPLICATION LIFESPAN (Startup/Shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs on startup and shutdown.
    """
    # === STARTUP ===
    logger.info("=" * 50)
    logger.info(f"🚀 Starting {settings.APP_NAME}")
    logger.info(f"   Environment: {settings.APP_ENV}")
    logger.info(f"   Debug: {settings.DEBUG}")
    logger.info("=" * 50)
    
    # Verify database connection
    try:
        from database import check_database_connection
        # Commented out for now - enable when ready
        # connected = await check_database_connection()
        # if not connected:
        #     logger.error("Database connection failed!")
        logger.info("✅ Database configuration loaded")
    except Exception as e:
        logger.warning(f"Database check skipped: {e}")
    
    # Ensure storage directories exist
    try:
        from utils import file_manager
        logger.info(f"✅ Storage directories ready")
        logger.info(f"   Uploads: {settings.UPLOAD_PATH}")
        logger.info(f"   Processed: {settings.PROCESSED_PATH}")
        logger.info(f"   Exports: {settings.EXPORT_PATH}")
    except Exception as e:
        logger.warning(f"Storage check skipped: {e}")
    
    logger.info("✅ Application startup complete")
    logger.info(f"📖 API Docs: http://localhost:{settings.BACKEND_PORT}/docs")
    
    yield  # Application runs here
    
    # === SHUTDOWN ===
    logger.info("Shutting down application...")
    
    # Cleanup database connections
    try:
        from database import async_dispose_engines
        await async_dispose_engines()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.warning(f"Database cleanup skipped: {e}")
    
    logger.info("👋 Application shutdown complete")


# =============================================================================
# OPENAPI TAGS DOCUMENTATION
# =============================================================================

tags_metadata = [
    {
        "name": "Health",
        "description": "Health check and system status endpoints.",
    },
    {
        "name": "Documents",
        "description": "Upload, list, retrieve, and delete documents. Supports PNG, JPG, JPEG, and PDF formats.",
    },
    {
        "name": "Extractions",
        "description": "OCR extraction operations including starting extraction, retrieving results, and updating fields.",
    },
    {
        "name": "Exports",
        "description": "Export extracted data to various formats: Excel, JSON, CSV, PDF.",
    },
    {
        "name": "Statistics",
        "description": "Dashboard statistics and analytics endpoints.",
    },
]


# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## 📄 OCR Form Data Extraction API
    
    Extract structured key-value data from forms using AI-powered OCR.
    
    ### Features
    - **Upload** documents (PNG, JPG, PDF)
    - **Extract** data with Azure Document Intelligence + Gemini AI
    - **Edit** extracted fields with confidence indicators
    - **Export** to Excel, JSON, CSV, PDF
    
    ### Workflow
    1. Upload a document
    2. Process with OCR + LLM extraction
    3. Review and edit extracted fields
    4. Finalize and save to database
    5. Export data in your preferred format
    
    ### Rate Limits
    - General API: 60 requests/minute
    - OCR endpoints: 20 requests/minute
    - LLM endpoints: 30 requests/minute
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)


# =============================================================================
# MIDDLEWARE
# =============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting Middleware (60 requests per minute per client)
try:
    from utils.rate_limit import RateLimitMiddleware
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=60,
        requests_per_hour=1000,
        exclude_paths=["/health", "/docs", "/redoc", "/openapi.json", "/"]
    )
    logger.info("✅ Rate limiting middleware enabled")
except ImportError:
    logger.warning("⚠️ Rate limiting middleware not available")


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header to all responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with clear messages"""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": errors
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.exception(f"Unhandled error: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


# =============================================================================
# ROOT ENDPOINTS
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API info"""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns status of all system components.
    """
    health = {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "environment": settings.APP_ENV,
        "debug": settings.DEBUG,
        "checks": {}
    }
    
    # Check database connection
    try:
        from database import sync_engine
        with sync_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health["checks"]["database"] = {
            "status": "ok",
            "type": "postgresql"
        }
    except Exception as e:
        health["checks"]["database"] = {
            "status": "error",
            "message": str(e)
        }
        health["status"] = "degraded"
    
    # Check storage directories
    try:
        storage_checks = {
            "uploads": settings.UPLOAD_PATH.exists(),
            "processed": settings.PROCESSED_PATH.exists(),
            "exports": settings.EXPORT_PATH.exists()
        }
        all_ok = all(storage_checks.values())
        health["checks"]["storage"] = {
            "status": "ok" if all_ok else "warning",
            "directories": storage_checks
        }
    except Exception as e:
        health["checks"]["storage"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Check Gemini API configuration
    try:
        gemini_configured = bool(settings.GEMINI_API_KEY)
        health["checks"]["gemini"] = {
            "status": "ok" if gemini_configured else "warning",
            "configured": gemini_configured,
            "model": settings.GEMINI_MODEL
        }
        if not gemini_configured:
            health["status"] = "degraded"
    except Exception as e:
        health["checks"]["gemini"] = {
            "status": "error",
            "message": str(e)
        }
    
    return health


# =============================================================================
# INCLUDE ROUTERS
# =============================================================================

from api.router import api_router
app.include_router(api_router, prefix="/api")


# =============================================================================
# DEV ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.is_development,
        log_level=settings.LOG_LEVEL.lower()
    )
