"""
File Manager Utility
====================
Handles all file operations: upload, storage, retrieval, deletion.

Features:
    - Organized storage structure (by date)
    - Unique filename generation (prevents overwrites)
    - File validation (size, type)
    - Safe file operations
"""

import os
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, BinaryIO
from uuid import uuid4
import mimetypes

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings


class FileManager:
    """
    Centralized file management for the OCR system.
    
    Usage:
        fm = FileManager()
        path, filename = fm.save_upload(file, "document.pdf")
        fm.delete_file(path)
    """
    
    # File type mappings
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
    MIME_TYPES = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'pdf': 'application/pdf'
    }
    
    def __init__(self):
        self.upload_dir = settings.UPLOAD_PATH
        self.processed_dir = settings.PROCESSED_PATH
        self.export_dir = settings.EXPORT_PATH
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist"""
        for directory in [self.upload_dir, self.processed_dir, self.export_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # FILE VALIDATION
    # =========================================================================
    
    def get_extension(self, filename: str) -> str:
        """Extract lowercase file extension"""
        return Path(filename).suffix.lower().lstrip('.')
    
    def is_allowed_extension(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        ext = self.get_extension(filename)
        return ext in self.ALLOWED_EXTENSIONS
    
    def validate_file(
        self, 
        filename: str, 
        file_size: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate file for upload.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check extension
        if not self.is_allowed_extension(filename):
            ext = self.get_extension(filename)
            allowed = ', '.join(self.ALLOWED_EXTENSIONS)
            return False, f"File type '{ext}' not allowed. Allowed: {allowed}"
        
        # Check size
        max_size = settings.max_upload_bytes
        if file_size > max_size:
            max_mb = settings.MAX_UPLOAD_SIZE_MB
            size_mb = file_size / (1024 * 1024)
            return False, f"File too large ({size_mb:.1f}MB). Maximum: {max_mb}MB"
        
        return True, None
    
    def get_mime_type(self, filename: str) -> str:
        """Get MIME type from filename"""
        ext = self.get_extension(filename)
        return self.MIME_TYPES.get(ext, 'application/octet-stream')
    
    # =========================================================================
    # FILE NAMING
    # =========================================================================
    
    def generate_unique_filename(
        self, 
        original_filename: str,
        prefix: Optional[str] = None
    ) -> str:
        """
        Generate unique filename while preserving extension.
        
        Format: {prefix}_{timestamp}_{short_uuid}.{ext}
        Example: doc_20240115_a1b2c3d4.pdf
        """
        ext = self.get_extension(original_filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid4())[:8]
        
        prefix = prefix or "doc"
        # Sanitize prefix
        prefix = "".join(c for c in prefix if c.isalnum() or c in "_-")[:20]
        
        return f"{prefix}_{timestamp}_{short_uuid}.{ext}"
    
    def sanitize_filename(self, filename: str) -> str:
        """Remove dangerous characters from filename"""
        # Keep only safe characters
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        name = Path(filename).stem
        ext = self.get_extension(filename)
        
        sanitized = "".join(c if c in safe_chars else "_" for c in name)
        sanitized = sanitized[:100]  # Limit length
        
        return f"{sanitized}.{ext}" if sanitized else f"file.{ext}"
    
    # =========================================================================
    # STORAGE PATHS
    # =========================================================================
    
    def get_date_subdir(self, base_dir: Path) -> Path:
        """Get date-based subdirectory (YYYY/MM)"""
        now = datetime.now()
        subdir = base_dir / str(now.year) / f"{now.month:02d}"
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir
    
    def get_upload_path(self, filename: str) -> Path:
        """Get full path for uploaded file"""
        subdir = self.get_date_subdir(self.upload_dir)
        return subdir / filename
    
    def get_processed_path(self, filename: str) -> Path:
        """Get full path for processed file"""
        subdir = self.get_date_subdir(self.processed_dir)
        return subdir / filename
    
    def get_export_path(self, filename: str) -> Path:
        """Get full path for export file"""
        subdir = self.get_date_subdir(self.export_dir)
        return subdir / filename
    
    def get_relative_path(self, full_path: Path) -> str:
        """Convert absolute path to relative (from project root)"""
        try:
            return str(full_path.relative_to(settings.PROJECT_ROOT))
        except ValueError:
            return str(full_path)
    
    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================
    
    def save_upload(
        self, 
        file_content: bytes,
        original_filename: str,
        custom_filename: Optional[str] = None
    ) -> Tuple[Path, str]:
        """
        Save uploaded file to storage.
        
        Args:
            file_content: File bytes
            original_filename: Original uploaded filename
            custom_filename: Optional custom name (will be sanitized)
        
        Returns:
            Tuple of (full_path, stored_filename)
        """
        # Generate unique filename
        if custom_filename:
            base_name = self.sanitize_filename(custom_filename)
            ext = self.get_extension(original_filename)
            base_name = Path(base_name).stem + f".{ext}"
        else:
            base_name = Path(original_filename).stem
        
        unique_filename = self.generate_unique_filename(original_filename, base_name[:20])
        
        # Get storage path
        file_path = self.get_upload_path(unique_filename)
        
        # Write file
        file_path.write_bytes(file_content)
        
        return file_path, unique_filename
    
    def save_upload_stream(
        self, 
        file_stream: BinaryIO,
        original_filename: str,
        chunk_size: int = 8192
    ) -> Tuple[Path, str, int]:
        """
        Save uploaded file from stream (for large files).
        
        Returns:
            Tuple of (full_path, stored_filename, file_size)
        """
        unique_filename = self.generate_unique_filename(original_filename)
        file_path = self.get_upload_path(unique_filename)
        
        total_size = 0
        with open(file_path, 'wb') as dest:
            while chunk := file_stream.read(chunk_size):
                dest.write(chunk)
                total_size += len(chunk)
        
        return file_path, unique_filename, total_size
    
    def save_processed(
        self, 
        content: str | bytes,
        filename: str,
        extension: str = "json"
    ) -> Path:
        """
        Save processed output (OCR results, etc.)
        
        Args:
            content: String or bytes content
            filename: Base filename (without extension)
            extension: File extension
        
        Returns:
            Full path to saved file
        """
        full_filename = f"{filename}.{extension}"
        file_path = self.get_processed_path(full_filename)
        
        if isinstance(content, str):
            file_path.write_text(content, encoding='utf-8')
        else:
            file_path.write_bytes(content)
        
        return file_path
    
    def save_export(
        self, 
        content: bytes,
        filename: str
    ) -> Path:
        """
        Save export file.
        
        Args:
            content: File bytes
            filename: Full filename with extension
        
        Returns:
            Full path to saved file
        """
        file_path = self.get_export_path(filename)
        file_path.write_bytes(content)
        return file_path
    
    def read_file(self, file_path: Path | str) -> bytes:
        """Read file content as bytes"""
        path = Path(file_path) if isinstance(file_path, str) else file_path
        return path.read_bytes()
    
    def read_text(self, file_path: Path | str) -> str:
        """Read file content as text"""
        path = Path(file_path) if isinstance(file_path, str) else file_path
        return path.read_text(encoding='utf-8')
    
    def delete_file(self, file_path: Path | str) -> bool:
        """
        Delete file safely.
        
        Returns:
            True if deleted, False if not found
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        
        if path.exists():
            path.unlink()
            return True
        return False
    
    def copy_file(self, source: Path, dest_dir: Path) -> Path:
        """Copy file to destination directory"""
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / source.name
        shutil.copy2(source, dest_path)
        return dest_path
    
    def move_file(self, source: Path, dest_dir: Path) -> Path:
        """Move file to destination directory"""
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / source.name
        shutil.move(str(source), str(dest_path))
        return dest_path
    
    def file_exists(self, file_path: Path | str) -> bool:
        """Check if file exists"""
        path = Path(file_path) if isinstance(file_path, str) else file_path
        return path.exists()
    
    def get_file_size(self, file_path: Path | str) -> int:
        """Get file size in bytes"""
        path = Path(file_path) if isinstance(file_path, str) else file_path
        return path.stat().st_size if path.exists() else 0
    
    def get_file_hash(self, file_path: Path | str, algorithm: str = 'md5') -> str:
        """Calculate file hash"""
        path = Path(file_path) if isinstance(file_path, str) else file_path
        
        hasher = hashlib.new(algorithm)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()


# Singleton instance
file_manager = FileManager()
