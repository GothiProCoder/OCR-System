"""
OCR Service - PaddleOCR-VL Implementation
==========================================
Complete rewrite using PaddleOCR-VL for document parsing.

Features:
- Page-level document parsing with layout detection
- Element-level recognition (text, tables, formulas, charts)
- 109 language support
- Native Markdown/JSON/HTML output
- CPU and GPU support
- ~2.5GB VRAM (vs Chandra's 8GB+)

Architecture:
- PP-DocLayoutV2: Layout detection and reading order
- PaddleOCR-VL-0.9B: Vision-Language model for recognition

Note: Requires NumPy < 2.0 for compatibility with PaddleOCR-VL.
Run: pip install "numpy<2.0.0" if you get scalar conversion errors.
"""

from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from PIL import Image
from dataclasses import dataclass, field
import time
import logging
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import os
import tempfile
import io

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from utils.image_preprocessing import image_preprocessor

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OCROutput:
    """Result from OCR processing."""
    markdown: str = ""
    html: str = ""
    json_output: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    page_number: int = 1
    image_width: int = 0
    image_height: int = 0
    layout_boxes: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "markdown": self.markdown,
            "html": self.html,
            "json_output": self.json_output,
            "processing_time_ms": self.processing_time_ms,
            "success": self.success,
            "error": self.error,
            "page_number": self.page_number,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "layout_boxes": self.layout_boxes
        }


@dataclass
class DocumentOCRResult:
    """Complete OCR result for a document."""
    pages: List[OCROutput] = field(default_factory=list)
    total_pages: int = 0
    total_processing_time_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    combined_markdown: str = ""
    combined_html: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages": [p.to_dict() for p in self.pages],
            "total_pages": self.total_pages,
            "total_processing_time_ms": self.total_processing_time_ms,
            "success": self.success,
            "error": self.error,
            "combined_markdown": self.combined_markdown,
            "combined_html": self.combined_html
        }


# =============================================================================
# OCR SERVICE - PaddleOCR-VL Implementation
# =============================================================================

class OCRService:
    """
    PaddleOCR-VL Document Parsing Service.
    
    Features:
    - Full page-level document parsing with layout detection
    - Element-level recognition (text, tables, formulas, charts)
    - Native Markdown/JSON/HTML output
    - CPU and GPU support (auto-detect)
    - Lazy model loading (loads on first use)
    
    Model Info:
    - PaddleOCR-VL-0.9B: 0.9 billion parameters
    - VRAM requirement: ~2.5GB GPU or ~5GB RAM for CPU
    - Inference time: 2-3 seconds/page on GPU, 30-60 seconds on CPU
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        
        self._pipeline = None
        self._model_loaded = False
        self._model_lock = threading.Lock()
        self._device = None
        self._is_gpu = False
        
        # Single worker for sequential processing
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._semaphore = threading.Semaphore(1)
        
        # Configuration from settings
        self.max_dimension = getattr(settings, 'OCR_MAX_IMAGE_DIMENSION', 2000)
        self.use_layout_detection = getattr(settings, 'PADDLEOCR_USE_LAYOUT_DETECTION', True)
        self.use_doc_orientation_classify = getattr(settings, 'PADDLEOCR_USE_DOC_ORIENTATION', False)
        self.use_doc_unwarping = getattr(settings, 'PADDLEOCR_USE_DOC_UNWARPING', False)
        self.use_chart_recognition = getattr(settings, 'PADDLEOCR_USE_CHART_RECOGNITION', False)
        self._configured_device = getattr(settings, 'PADDLEOCR_DEVICE', '') or None
        
        self._initialized = True
        logger.info("OCR Service initialized (PaddleOCR-VL mode)")
    
    # =========================================================================
    # HARDWARE DETECTION
    # =========================================================================
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware (CPU/GPU)."""
        # Use configured device if specified
        if self._configured_device:
            is_gpu = self._configured_device.startswith('gpu')
            return {
                'gpu_available': is_gpu,
                'gpu_count': 1 if is_gpu else 0,
                'device': self._configured_device
            }
        
        # Auto-detect hardware
        try:
            import paddle
            gpu_available = paddle.is_compiled_with_cuda()
            
            if gpu_available:
                try:
                    gpu_count = paddle.device.cuda.device_count()
                except:
                    gpu_count = 0
                    gpu_available = False
            else:
                gpu_count = 0
            
            return {
                'gpu_available': gpu_available,
                'gpu_count': gpu_count,
                'device': 'gpu:0' if gpu_available else 'cpu'
            }
        except ImportError:
            logger.warning("PaddlePaddle not imported yet, assuming CPU")
            return {
                'gpu_available': False,
                'gpu_count': 0,
                'device': 'cpu'
            }
    
    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    
    def _ensure_pipeline_loaded(self) -> None:
        """Lazy-load PaddleOCR-VL pipeline on first use."""
        if self._model_loaded:
            return
        
        with self._model_lock:
            if self._model_loaded:
                return
            
            logger.info("=" * 60)
            logger.info("LOADING PaddleOCR-VL Pipeline")
            logger.info("This may take a few minutes on first run (downloading models)...")
            logger.info("=" * 60)
            
            start_time = time.time()
            
            try:
                from paddleocr import PaddleOCRVL
                
                # Detect hardware
                hw = self._detect_hardware()
                self._device = hw['device']
                self._is_gpu = hw['gpu_available']
                
                logger.info(f"Device: {self._device}")
                logger.info(f"GPU Available: {self._is_gpu}")
                
                # Initialize pipeline with configured settings
                self._pipeline = PaddleOCRVL(
                    device=self._device,
                    use_layout_detection=self.use_layout_detection,
                    use_doc_orientation_classify=self.use_doc_orientation_classify,
                    use_doc_unwarping=self.use_doc_unwarping,
                    use_chart_recognition=self.use_chart_recognition,
                )
                
                elapsed = time.time() - start_time
                self._model_loaded = True
                
                logger.info(f"PaddleOCR-VL loaded successfully in {elapsed:.1f}s")
                logger.info("=" * 60)
                
            except ImportError as e:
                logger.error(f"PaddleOCR not installed: {e}")
                logger.error("Please install: pip install paddleocr[doc-parser]")
                raise RuntimeError(f"PaddleOCR not installed: {e}")
            
            except Exception as e:
                logger.error(f"Failed to load PaddleOCR-VL: {e}")
                raise RuntimeError(f"Failed to load PaddleOCR-VL: {e}")
    
    def _unload_pipeline(self) -> None:
        """Unload pipeline to free memory (optional)."""
        with self._model_lock:
            if self._pipeline is not None:
                logger.info("Unloading PaddleOCR-VL pipeline...")
                del self._pipeline
                self._pipeline = None
                self._model_loaded = False
                gc.collect()
                logger.info("Pipeline unloaded")
    
    # =========================================================================
    # CORE PROCESSING
    # =========================================================================
    
    def _process_with_pipeline(self, input_path: str) -> List[Dict[str, Any]]:
        """
        Run PaddleOCR-VL pipeline on input.
        
        Args:
            input_path: Path to image or PDF file
            
        Returns:
            List of result dictionaries (one per page)
        """
        self._ensure_pipeline_loaded()
        
        results = []
        
        try:
            output = self._pipeline.predict(input_path)
            
            for res in output:
                # Extract markdown
                md_info = res.markdown
                markdown_text = md_info.get("markdown_text", "") if isinstance(md_info, dict) else ""
                
                # Extract JSON
                json_data = res.json if hasattr(res, 'json') else {}
                
                # Extract layout boxes if available
                layout_boxes = []
                if hasattr(res, 'json') and isinstance(res.json, dict):
                    layout_det = res.json.get('layout_det_res', {})
                    if isinstance(layout_det, dict):
                        boxes = layout_det.get('boxes', [])
                        for box in boxes:
                            layout_boxes.append({
                                'label': box.get('label', ''),
                                'score': float(box.get('score', 0)),
                                'coordinate': [float(c) for c in box.get('coordinate', [])]
                            })
                
                # Generate HTML (save to temp and read)
                html_content = self._generate_html_from_result(res)
                
                results.append({
                    'markdown': markdown_text,
                    'html': html_content,
                    'json': json_data,
                    'layout_boxes': layout_boxes,
                    'error': None
                })
                
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            results.append({
                'markdown': '',
                'html': '',
                'json': {},
                'layout_boxes': [],
                'error': str(e)
            })
        
        return results
    
    def _generate_html_from_result(self, res) -> str:
        """Generate HTML from PaddleOCR result."""
        try:
            # Create temp directory for HTML output
            with tempfile.TemporaryDirectory() as tmpdir:
                res.save_to_html(save_path=tmpdir)
                
                # Find the generated HTML file
                html_files = list(Path(tmpdir).glob("*.html"))
                if html_files:
                    with open(html_files[0], 'r', encoding='utf-8') as f:
                        return f.read()
            
            return ""
        except Exception as e:
            logger.warning(f"HTML generation failed: {e}")
            return ""
    
    # =========================================================================
    # IMAGE PROCESSING
    # =========================================================================
    
    def _process_single_image_sync(
        self,
        image: Image.Image,
        page_number: int = 1
    ) -> OCROutput:
        """Process a single image synchronously."""
        with self._semaphore:
            start_time = time.time()
            original_size = image.size
            
            try:
                # Resize if needed for performance
                optimized = image_preprocessor.resize_if_needed(
                    image, 
                    max_dimension=self.max_dimension
                )
                
                # Save to temp file (PaddleOCR needs file path)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                    optimized.save(tmp_path, 'PNG')
                
                try:
                    logger.info(f"Processing image page {page_number} ({optimized.size})...")
                    
                    results = self._process_with_pipeline(tmp_path)
                    
                    processing_time = int((time.time() - start_time) * 1000)
                    
                    if results and not results[0].get('error'):
                        result = results[0]
                        return OCROutput(
                            markdown=result['markdown'],
                            html=result['html'],
                            json_output=result['json'],
                            processing_time_ms=processing_time,
                            success=True,
                            page_number=page_number,
                            image_width=original_size[0],
                            image_height=original_size[1],
                            layout_boxes=result.get('layout_boxes', [])
                        )
                    else:
                        error = results[0].get('error') if results else "No results"
                        return OCROutput(
                            success=False,
                            error=error,
                            processing_time_ms=processing_time,
                            page_number=page_number,
                            image_width=original_size[0],
                            image_height=original_size[1]
                        )
                        
                finally:
                    # Cleanup temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                
            except Exception as e:
                processing_time = int((time.time() - start_time) * 1000)
                logger.error(f"OCR failed: {e}", exc_info=True)
                
                return OCROutput(
                    success=False,
                    error=str(e),
                    processing_time_ms=processing_time,
                    page_number=page_number,
                    image_width=original_size[0],
                    image_height=original_size[1]
                )
    
    def process_image_sync(
        self,
        image_source: Union[str, Path, Image.Image, bytes],
        page_number: int = 1
    ) -> OCROutput:
        """
        Process image from various sources synchronously.
        
        Args:
            image_source: Image path, PIL Image, or bytes
            page_number: Page number for this image
            
        Returns:
            OCROutput with results
        """
        # Load image from various sources
        if isinstance(image_source, bytes):
            image = image_preprocessor.load_image_bytes(image_source)
        elif isinstance(image_source, (str, Path)):
            image = image_preprocessor.load_image(image_source)
        elif isinstance(image_source, Image.Image):
            image = image_source
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")
        
        # Preprocess image
        image = image_preprocessor.optimize_for_ocr(
            image,
            apply_contrast=True,
            apply_sharpness=True,
            apply_denoise=False
        )
        
        return self._process_single_image_sync(image, page_number)
    
    # =========================================================================
    # PDF PROCESSING
    # =========================================================================
    
    def process_pdf_sync(self, pdf_path: Union[str, Path]) -> DocumentOCRResult:
        """
        Process a PDF document synchronously.
        
        PaddleOCR-VL can process PDFs directly, handling multi-page documents
        and concatenating results automatically.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DocumentOCRResult with all pages
        """
        start_time = time.time()
        pdf_path = Path(pdf_path)
        
        try:
            if not pdf_path.exists():
                return DocumentOCRResult(
                    success=False,
                    error=f"File not found: {pdf_path}"
                )
            
            self._ensure_pipeline_loaded()
            
            logger.info(f"Processing PDF: {pdf_path.name}")
            
            # Process PDF directly with PaddleOCR
            output = self._pipeline.predict(str(pdf_path))
            
            page_results: List[OCROutput] = []
            markdown_list = []
            
            page_num = 0
            for res in output:
                page_num += 1
                page_start = time.time()
                
                # Extract markdown
                md_info = res.markdown
                markdown_text = md_info.get("markdown_text", "") if isinstance(md_info, dict) else ""
                markdown_list.append(md_info)
                
                # Extract JSON
                json_data = res.json if hasattr(res, 'json') else {}
                
                # Extract layout boxes
                layout_boxes = []
                if hasattr(res, 'json') and isinstance(res.json, dict):
                    layout_det = res.json.get('layout_det_res', {})
                    if isinstance(layout_det, dict):
                        boxes = layout_det.get('boxes', [])
                        for box in boxes:
                            layout_boxes.append({
                                'label': box.get('label', ''),
                                'score': float(box.get('score', 0)),
                                'coordinate': [float(c) for c in box.get('coordinate', [])]
                            })
                
                # Generate HTML
                html_content = self._generate_html_from_result(res)
                
                page_time = int((time.time() - page_start) * 1000)
                
                page_results.append(OCROutput(
                    markdown=markdown_text,
                    html=html_content,
                    json_output=json_data,
                    processing_time_ms=page_time,
                    success=True,
                    page_number=page_num,
                    layout_boxes=layout_boxes
                ))
                
                logger.info(f"  Page {page_num} processed ({page_time}ms)")
            
            # Concatenate all markdown pages
            combined_md = self._pipeline.concatenate_markdown_pages(markdown_list)
            combined_html = self._combine_html(page_results)
            
            total_time = int((time.time() - start_time) * 1000)
            all_success = all(p.success for p in page_results)
            
            return DocumentOCRResult(
                pages=page_results,
                total_pages=len(page_results),
                total_processing_time_ms=total_time,
                success=all_success,
                error=None if all_success else "Some pages failed",
                combined_markdown=combined_md,
                combined_html=combined_html
            )
            
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            logger.error(f"PDF processing failed: {e}", exc_info=True)
            return DocumentOCRResult(
                success=False,
                error=str(e),
                total_processing_time_ms=total_time
            )
    
    def process_pdf_as_images_sync(self, pdf_path: Union[str, Path]) -> DocumentOCRResult:
        """
        Alternative: Process PDF by converting to images first.
        
        This can be more reliable for some PDFs but is slower.
        Use process_pdf_sync() for better performance.
        """
        start_time = time.time()
        
        try:
            images = image_preprocessor.pdf_to_images(pdf_path)
            
            if not images:
                return DocumentOCRResult(success=False, error="No pages found in PDF")
            
            page_results: List[OCROutput] = []
            
            for i, page_image in enumerate(images, start=1):
                logger.info(f"Processing PDF page {i}/{len(images)}")
                result = self._process_single_image_sync(page_image, i)
                page_results.append(result)
                
                # Free memory
                del page_image
                gc.collect()
            
            del images
            
            combined_md = self._combine_markdown(page_results)
            combined_html = self._combine_html(page_results)
            
            total_time = int((time.time() - start_time) * 1000)
            all_success = all(p.success for p in page_results)
            
            return DocumentOCRResult(
                pages=page_results,
                total_pages=len(page_results),
                total_processing_time_ms=total_time,
                success=all_success,
                error=None if all_success else "Some pages failed",
                combined_markdown=combined_md,
                combined_html=combined_html
            )
            
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            logger.error(f"PDF processing failed: {e}")
            return DocumentOCRResult(
                success=False,
                error=str(e),
                total_processing_time_ms=total_time
            )
    
    # =========================================================================
    # ASYNC METHODS
    # =========================================================================
    
    async def process_image(
        self,
        image_source: Union[str, Path, Image.Image, bytes],
        page_number: int = 1,
        timeout: float = 900.0  # 15 min for slow CPU
    ) -> OCROutput:
        """Async wrapper for image processing."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.process_image_sync, image_source, page_number),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return OCROutput(success=False, error=f"Timed out after {timeout}s")
    
    async def process_pdf(
        self,
        pdf_path: Union[str, Path],
        timeout: float = 3600.0  # 1 hour for multi-page PDFs
    ) -> DocumentOCRResult:
        """Async wrapper for PDF processing."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.process_pdf_sync, pdf_path),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return DocumentOCRResult(success=False, error=f"Timed out after {timeout}s")
    
    async def process_document(
        self,
        file_path: Union[str, Path],
        file_type: str
    ) -> DocumentOCRResult:
        """
        Process any supported document type.
        
        Args:
            file_path: Path to document
            file_type: File extension (pdf, png, jpg, jpeg)
            
        Returns:
            DocumentOCRResult with all pages
        """
        file_type = file_type.lower().strip('.')
        path = Path(file_path)
        
        if not path.exists():
            return DocumentOCRResult(success=False, error=f"File not found: {path}")
        
        if file_type == 'pdf':
            return await self.process_pdf(path)
        elif file_type in ('png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'):
            result = await self.process_image(path)
            return DocumentOCRResult(
                pages=[result],
                total_pages=1,
                total_processing_time_ms=result.processing_time_ms,
                success=result.success,
                error=result.error,
                combined_markdown=result.markdown,
                combined_html=result.html
            )
        else:
            return DocumentOCRResult(success=False, error=f"Unsupported file type: {file_type}")
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _combine_markdown(self, pages: List[OCROutput]) -> str:
        """Combine markdown from multiple pages."""
        sections = []
        for page in pages:
            if page.markdown:
                if len(pages) > 1:
                    sections.append(f"## Page {page.page_number}\n\n{page.markdown}")
                else:
                    sections.append(page.markdown)
        return "\n\n---\n\n".join(sections)
    
    def _combine_html(self, pages: List[OCROutput]) -> str:
        """Combine HTML from multiple pages."""
        sections = []
        for page in pages:
            if page.html:
                if len(pages) > 1:
                    sections.append(f'<section data-page="{page.page_number}">\n{page.html}\n</section>')
                else:
                    sections.append(page.html)
        return "\n<hr>\n".join(sections)
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            "model_loaded": self._model_loaded,
            "model_name": "PaddleOCR-VL-0.9B",
            "mode": "gpu" if self._is_gpu else "cpu",
            "device": self._device or "not_loaded",
            "use_layout_detection": self.use_layout_detection,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_chart_recognition": self.use_chart_recognition,
            "max_dimension": self.max_dimension,
            "engine": "PaddleOCR-VL"
        }
    
    def preload_model(self) -> None:
        """Preload model at startup."""
        self._ensure_pipeline_loaded()
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def cleanup(self) -> None:
        """Full cleanup."""
        self._unload_pipeline()
        if self._executor:
            self._executor.shutdown(wait=False)
        logger.info("OCR Service cleaned up")


# =============================================================================
# SINGLETON & EXPORTS
# =============================================================================

ocr_service = OCRService()


async def ocr_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for OCR processing."""
    document_path = state.get("document_path")
    file_type = state.get("file_type", "")
    
    if not document_path:
        return {
            **state,
            "ocr_result": None,
            "ocr_markdown": "",
            "ocr_success": False,
            "ocr_error": "No document_path in state",
            "ocr_time_ms": 0
        }
    
    result = await ocr_service.process_document(document_path, file_type)
    
    return {
        **state,
        "ocr_result": result.to_dict(),
        "ocr_markdown": result.combined_markdown,
        "ocr_success": result.success,
        "ocr_error": result.error,
        "ocr_time_ms": result.total_processing_time_ms
    }


def preload_ocr_model() -> None:
    """Preload model at startup."""
    try:
        ocr_service.preload_model()
    except Exception as e:
        logger.warning(f"OCR preload failed: {e}")


async def get_ocr_status() -> Dict[str, Any]:
    """Get service status."""
    return ocr_service.get_status()
