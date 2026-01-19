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
import numpy as np

# OpenCV for deskewing - optional dependency
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

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
    
    # =========================================================================
    # AZURE DOCUMENT INTELLIGENCE PREPROCESSING
    # =========================================================================
    
    def deskew(self, image: Image.Image) -> Tuple[Image.Image, float]:
        """
        Automatically detect and correct SMALL skew in image.
        Uses Hough Line Transform to detect dominant line angles.
        
        Args:
            image: PIL Image to deskew
            
        Returns:
            Tuple of (deskewed_image, angle_corrected)
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, skipping deskew")
            return image, 0.0
        
        logger.info("Deskewing image...")
        
        # Convert PIL to OpenCV format
        if image.mode == 'L':
            cv_image = np.array(image)
            gray = cv_image
        else:
            cv_image = np.array(image.convert('RGB'))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=10
        )
        
        if lines is None:
            logger.info("No lines detected, skipping deskew")
            return image, 0.0
        
        # Calculate angles of all detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Normalize to -45 to 45 range (small corrections only!)
            if angle < -45:
                angle = angle + 90
            elif angle > 45:
                angle = angle - 90
                
            angles.append(angle)
        
        # Get median angle (robust to outliers)
        angle = float(np.median(angles))
        
        logger.info(f"Detected skew angle: {angle:.2f}°")
        
        # Only rotate if angle is significant but SMALL
        if abs(angle) < 0.5:
            logger.info("Image already aligned (angle < 0.5°)")
            return image, angle
        
        if abs(angle) > 45:
            logger.warning(f"Angle too large ({angle:.2f}°), probably wrong detection. Skipping.")
            return image, 0.0
        
        # Rotate image to deskew
        (h, w) = cv_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        deskewed = cv2.warpAffine(
            cv_image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Convert back to PIL
        if image.mode == 'L':
            result = Image.fromarray(deskewed)
        else:
            deskewed_rgb = cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB)
            result = Image.fromarray(deskewed_rgb)
        
        logger.info(f"Image deskewed by {angle:.2f}°")
        return result, angle
    
    def adaptive_binarize(self, image: Image.Image) -> Image.Image:
        """
        Adaptive thresholding for better text extraction on forms.
        Better than fixed threshold for documents with varied lighting.
        
        Args:
            image: PIL Image to binarize
            
        Returns:
            Binarized PIL Image
        """
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, using simple binarize")
            return self.binarize(image)
        
        logger.info("Applying adaptive binarization...")
        
        # Convert to grayscale
        if image.mode != 'L':
            gray = np.array(image.convert('L'))
        else:
            gray = np.array(image)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
        return Image.fromarray(binary)
    
    def compress_for_azure(
        self,
        image: Image.Image,
        target_size_mb: float = 2.0,
        initial_quality: int = 95,
        min_quality: int = 30
    ) -> bytes:
        """
        Compress image to fit Azure Document Intelligence size limits.
        Uses iterative JPEG quality reduction if needed.
        
        Azure limit is 5MB, we target 2MB for safety margin.
        
        Args:
            image: PIL Image to compress
            target_size_mb: Target size in MB (default: 2.0)
            initial_quality: Starting JPEG quality (default: 95)
            min_quality: Minimum quality to try (default: 30)
            
        Returns:
            JPEG bytes under target size
        """
        target_bytes = int(target_size_mb * 1024 * 1024)
        
        # Ensure RGB mode for JPEG
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        elif image.mode == 'L':
            image = image.convert('RGB')
        
        quality = initial_quality
        
        while quality >= min_quality:
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality, optimize=True)
            size = buffer.tell()
            
            if size <= target_bytes:
                logger.info(f"Compressed to {size/1024/1024:.2f}MB at quality={quality}")
                return buffer.getvalue()
            
            logger.debug(f"Size {size/1024/1024:.2f}MB at quality={quality}, reducing...")
            quality -= 10
        
        # If still too large, also resize
        logger.warning("Quality reduction not enough, also resizing image")
        
        # Calculate scale factor to fit target
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=min_quality)
        current_size = buffer.tell()
        
        scale = (target_bytes / current_size) ** 0.5
        new_size = (int(image.width * scale), int(image.height * scale))
        
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        resized.save(buffer, format='JPEG', quality=min_quality, optimize=True)
        
        logger.info(f"Compressed to {buffer.tell()/1024/1024:.2f}MB after resize to {new_size}")
        return buffer.getvalue()
    
    def preprocess_for_azure(
        self,
        image: Union[str, Path, Image.Image, bytes],
        apply_deskew: bool = True,
        apply_binarize: bool = False,
        apply_contrast: bool = True,
        apply_sharpness: bool = True,
        target_size_mb: float = 2.0
    ) -> bytes:
        """
        Full preprocessing pipeline for Azure Document Intelligence.
        
        Pipeline:
        1. Load image
        2. Auto-orient (EXIF)
        3. Resize if too large
        4. Deskew (optional)
        5. Contrast enhancement (optional)
        6. Sharpness enhancement (optional)
        7. Binarize (optional, for noisy forms)
        8. Compress to target size
        
        Args:
            image: Path, PIL Image, or bytes
            apply_deskew: Correct skew angles (default: True)
            apply_binarize: Convert to binary (default: False)
            apply_contrast: Enhance contrast (default: True)
            apply_sharpness: Enhance sharpness (default: True)
            target_size_mb: Target size in MB (default: 2.0)
            
        Returns:
            JPEG bytes ready for Azure API
        """
        # Load image
        if isinstance(image, bytes):
            img = self.load_image_bytes(image)
        elif isinstance(image, (str, Path)):
            img = self.load_image(image)
        else:
            img = image
        
        logger.info(f"Preprocessing image: {img.size}, mode={img.mode}")
        
        # Auto-orient based on EXIF
        img = self.auto_orient(img)
        
        # Resize if too large (critical for performance)
        img = self.resize_if_needed(img)
        
        # Deskew
        if apply_deskew:
            img, angle = self.deskew(img)
        
        # Enhance contrast
        if apply_contrast and not apply_binarize:
            img = self.enhance_contrast(img, factor=1.2)
        
        # Sharpen
        if apply_sharpness and not apply_binarize:
            img = self.enhance_sharpness(img, factor=1.1)
        
        # Binarize (for noisy forms)
        if apply_binarize:
            img = self.adaptive_binarize(img)
        
        # Compress to Azure size limits
        compressed = self.compress_for_azure(img, target_size_mb=target_size_mb)
        
        logger.info(f"Preprocessing complete: {len(compressed)/1024:.1f}KB output")
        return compressed


# Singleton instance
image_preprocessor = ImagePreprocessor()
