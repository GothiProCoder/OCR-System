"""
Backend Configuration Module
============================
Centralized configuration using Pydantic Settings.
All settings are loaded from environment variables with sensible defaults.

Usage:
    from config import settings
    print(settings.DATABASE_URL)
    print(settings.UPLOAD_PATH)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field
from pathlib import Path
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses Pydantic Settings for validation and type coercion.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ===== APPLICATION =====
    APP_NAME: str = "FormExtract AI"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # ===== SERVER =====
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 8501
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"
    
    # ===== DATABASE =====
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    DB_NAME: str = "ocr_system"
    DATABASE_URL: str | None = None  # Can override full URL
    
    # ===== GEMINI AI =====
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    
    # ===== STORAGE =====
    UPLOAD_DIR: str = "storage/uploads"
    PROCESSED_DIR: str = "storage/processed"
    EXPORT_DIR: str = "storage/exports"
    
    # ===== FILE LIMITS =====
    MAX_UPLOAD_SIZE_MB: int = 20
    ALLOWED_EXTENSIONS: str = "png,jpg,jpeg,pdf"
    
    # ===== OCR SETTINGS =====
    OCR_MAX_IMAGE_DIMENSION: int = 2000
    OCR_INFERENCE_METHOD: str = "hf"
    
    # ===== EXTRACTION SETTINGS =====
    CONFIDENCE_HIGH_THRESHOLD: float = 0.85
    CONFIDENCE_MEDIUM_THRESHOLD: float = 0.60
    
    # ===== EXPORT SETTINGS =====
    EXPORT_PDF_AUTHOR: str = "FormExtract AI"
    EXPORT_EXCEL_SHEET_NAME: str = "Extracted Data"
    
    # ===== COMPUTED PROPERTIES =====
    
    @computed_field
    @property
    def PROJECT_ROOT(self) -> Path:
        """Project root directory (parent of backend folder)"""
        return Path(__file__).parent.parent.resolve()
    
    @computed_field
    @property
    def BACKEND_ROOT(self) -> Path:
        """Backend directory"""
        return Path(__file__).parent.resolve()
    
    @computed_field
    @property
    def database_url(self) -> str:
        """Construct database URL from components or use override"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @computed_field
    @property
    def async_database_url(self) -> str:
        """Async database URL for SQLAlchemy async engine"""
        base_url = self.database_url
        return base_url.replace("postgresql://", "postgresql+asyncpg://")
    
    @computed_field
    @property
    def UPLOAD_PATH(self) -> Path:
        """Absolute path to uploads directory"""
        path = self.PROJECT_ROOT / self.UPLOAD_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @computed_field
    @property
    def PROCESSED_PATH(self) -> Path:
        """Absolute path to processed files directory"""
        path = self.PROJECT_ROOT / self.PROCESSED_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @computed_field
    @property
    def EXPORT_PATH(self) -> Path:
        """Absolute path to exports directory"""
        path = self.PROJECT_ROOT / self.EXPORT_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @computed_field
    @property
    def allowed_extensions_list(self) -> List[str]:
        """List of allowed file extensions"""
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",")]
    
    @computed_field
    @property
    def allowed_origins_list(self) -> List[str]:
        """List of allowed CORS origins"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]
    
    @computed_field
    @property
    def max_upload_bytes(self) -> int:
        """Max upload size in bytes"""
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.APP_ENV == "development"
    
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.APP_ENV == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Use this function to access settings throughout the application.
    
    Example:
        from config import get_settings
        settings = get_settings()
    """
    return Settings()


# Convenience: Direct access to settings singleton
settings = get_settings()
