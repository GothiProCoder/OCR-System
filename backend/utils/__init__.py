"""
Utils Package
=============
Utility modules for file management and image processing.

Usage:
    from utils import file_manager, image_preprocessor
    
    # File operations
    path, name = file_manager.save_upload(content, "doc.pdf")
    
    # Image optimization
    optimized = image_preprocessor.optimize_for_ocr(image_path)
"""

from utils.file_manager import FileManager, file_manager
from utils.image_preprocessing import ImagePreprocessor, image_preprocessor


__all__ = [
    "FileManager",
    "file_manager",
    "ImagePreprocessor", 
    "image_preprocessor",
]
