"""
Image Preprocessing Utility
===========================
Optimizes images for OCR processing on CPU.

Features:
    - Resize large images (CPU optimization)
    - Deskew (straighten rotated scans)
    - Denoise (reduce noise for clearer text)
    - Contrast enhancement
    - PDF to image conversion
"""

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pathlib import Path
from typing import List, Tuple, Optional, Union
import io
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing for optimal OCR performance.
    
    Usage:
        preprocessor = ImagePreprocessor()
        optimized = preprocessor.optimize_for_ocr(image_path)
        pages = preprocessor.pdf_to_images(pdf_path)
    """
    
    def __init__(
        self,
        max_dimension: int = None,
        target_dpi: int = 300
    ):
        self.max_dimension = max_dimension or settings.OCR_MAX_IMAGE_DIMENSION
        self.target_dpi = target_dpi
    
    # =========================================================================
    # IMAGE LOADING
    # =========================================================================
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load image from path"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        image = Image.open(path)
        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        return image
    
    def load_image_bytes(self, image_bytes: bytes) -> Image.Image:
        """Load image from bytes"""
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        return image
    
    # =========================================================================
    # SIZE OPTIMIZATION (Critical for CPU performance)
    # =========================================================================
    
    def resize_if_needed(
        self, 
        image: Image.Image,
        max_dimension: int = None
    ) -> Image.Image:
        """
        Resize image if larger than max dimension.
        Preserves aspect ratio.
        
        This is the MOST IMPORTANT optimization for CPU inference.
        Processing a 4000x3000 image takes 4x longer than 2000x1500.
        """
        max_dim = max_dimension or self.max_dimension
        width, height = image.size
        
        if max(width, height) <= max_dim:
            return image
        
        # Calculate new size preserving aspect ratio
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        
        logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        
        # Use LANCZOS for high quality downscaling
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def get_optimal_size(
        self, 
        width: int, 
        height: int,
        max_dimension: int = None
    ) -> Tuple[int, int]:
        """Calculate optimal size for OCR"""
        max_dim = max_dimension or self.max_dimension
        
        if max(width, height) <= max_dim:
            return width, height
        
        if width > height:
            return max_dim, int(height * (max_dim / width))
        return int(width * (max_dim / height)), max_dim
    
    # =========================================================================
    # IMAGE ENHANCEMENT
    # =========================================================================
    
    def enhance_contrast(
        self, 
        image: Image.Image,
        factor: float = 1.3
    ) -> Image.Image:
        """
        Enhance image contrast for clearer text.
        
        Args:
            factor: 1.0 = no change, >1 = more contrast
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def enhance_sharpness(
        self, 
        image: Image.Image,
        factor: float = 1.2
    ) -> Image.Image:
        """
        Sharpen image for clearer text edges.
        
        Args:
            factor: 1.0 = no change, >1 = sharper
        """
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    def denoise(self, image: Image.Image) -> Image.Image:
        """
        Reduce noise in image.
        Uses median filter for salt-and-pepper noise.
        """
        return image.filter(ImageFilter.MedianFilter(size=3))
    
    def convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """Convert to grayscale (can improve OCR for some documents)"""
        return image.convert('L')
    
    def auto_orient(self, image: Image.Image) -> Image.Image:
        """Apply EXIF orientation if present"""
        return ImageOps.exif_transpose(image)
    
    def binarize(
        self, 
        image: Image.Image,
        threshold: int = 128
    ) -> Image.Image:
        """
        Convert to black and white (binary).
        Useful for very noisy documents.
        """
        grayscale = image.convert('L')
        return grayscale.point(lambda x: 255 if x > threshold else 0, '1')
    
    # =========================================================================
    # FULL OPTIMIZATION PIPELINE
    # =========================================================================
    
    def optimize_for_ocr(
        self,
        image: Union[str, Path, Image.Image, bytes],
        apply_contrast: bool = True,
        apply_sharpness: bool = True,
        apply_denoise: bool = False,
        grayscale: bool = False
    ) -> Image.Image:
        """
        Full optimization pipeline for OCR.
        
        Args:
            image: Path, Image object, or bytes
            apply_contrast: Enhance contrast
            apply_sharpness: Sharpen image
            apply_denoise: Apply noise reduction
            grayscale: Convert to grayscale
        
        Returns:
            Optimized PIL Image
        """
        # Load image
        if isinstance(image, bytes):
            img = self.load_image_bytes(image)
        elif isinstance(image, (str, Path)):
            img = self.load_image(image)
        else:
            img = image
        
        # Auto-orient based on EXIF
        img = self.auto_orient(img)
        
        # Resize if too large (CRITICAL for CPU performance)
        img = self.resize_if_needed(img)
        
        # Optional grayscale conversion
        if grayscale:
            img = self.convert_to_grayscale(img)
        
        # Optional denoise (before enhancement)
        if apply_denoise:
            img = self.denoise(img)
        
        # Enhance contrast
        if apply_contrast:
            img = self.enhance_contrast(img, factor=1.2)
        
        # Sharpen
        if apply_sharpness:
            img = self.enhance_sharpness(img, factor=1.1)
        
        return img
    
    # =========================================================================
    # PDF HANDLING
    # =========================================================================
    
    def pdf_to_images(
        self,
        pdf_path: Union[str, Path],
        dpi: int = None
    ) -> List[Image.Image]:
        """
        Convert PDF pages to images.
        
        Requires pdf2image library and poppler.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution (default: 300)
        
        Returns:
            List of PIL Images (one per page)
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "pdf2image not installed. Install with: pip install pdf2image\n"
                "Also install poppler: https://github.com/oschwartz10612/poppler-windows/releases"
            )
        
        dpi = dpi or self.target_dpi
        path = Path(pdf_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        
        logger.info(f"Converting PDF to images: {path} at {dpi} DPI")
        
        images = convert_from_path(
            str(path),
            dpi=dpi,
            fmt='png'
        )
        
        # Optimize each page
        optimized = []
        for i, img in enumerate(images):
            logger.debug(f"Optimizing page {i + 1}/{len(images)}")
            opt_img = self.resize_if_needed(img)
            optimized.append(opt_img)
        
        logger.info(f"Converted {len(optimized)} pages from PDF")
        return optimized
    
    def get_pdf_page_count(self, pdf_path: Union[str, Path]) -> int:
        """Get number of pages in PDF without full conversion"""
        try:
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(str(pdf_path))
            return info.get('Pages', 1)
        except Exception:
            # Fallback: do full conversion and count
            images = self.pdf_to_images(pdf_path)
            return len(images)
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def save_image(
        self,
        image: Image.Image,
        output_path: Union[str, Path],
        quality: int = 95,
        optimize: bool = True
    ) -> Path:
        """Save image to file"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with appropriate format
        if path.suffix.lower() in ('.jpg', '.jpeg'):
            image.save(path, 'JPEG', quality=quality, optimize=optimize)
        elif path.suffix.lower() == '.png':
            image.save(path, 'PNG', optimize=optimize)
        else:
            image.save(path)
        
        return path
    
    def image_to_bytes(
        self,
        image: Image.Image,
        format: str = 'PNG',
        quality: int = 95
    ) -> bytes:
        """Convert PIL Image to bytes"""
        buffer = io.BytesIO()
        
        if format.upper() in ('JPG', 'JPEG'):
            image.save(buffer, format='JPEG', quality=quality)
        else:
            image.save(buffer, format=format)
        
        return buffer.getvalue()
    
    def get_image_info(
        self, 
        image: Union[str, Path, Image.Image]
    ) -> dict:
        """Get image metadata"""
        if isinstance(image, (str, Path)):
            img = self.load_image(image)
        else:
            img = image
        
        return {
            'width': img.width,
            'height': img.height,
            'mode': img.mode,
            'format': img.format,
            'size_optimal': self.get_optimal_size(img.width, img.height),
            'needs_resize': max(img.width, img.height) > self.max_dimension
        }


# Singleton instance
image_preprocessor = ImagePreprocessor()
