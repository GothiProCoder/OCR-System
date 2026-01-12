"""
OCR Service - FINAL PRODUCTION VERSION
=======================================
Based on actual Chandra source code analysis:

CHANDRA INTERNALS (from .venv/Lib/site-packages/chandra/):
- Uses Qwen3VLForConditionalGeneration (NOT Qwen2VL)
- Uses Qwen3VLProcessor (NOT AutoProcessor)
- hf.py line 31: inputs.to("cuda") - HARDCODED CUDA
- load_model() uses device_map="auto"
- Settings.TORCH_DEVICE can override device

PROBLEM: 8GB RAM + CPU-only cannot use standard InferenceManager.
SOLUTION: Custom CPU loader with patched generate function.

Usage:
    from services.ocr_service import ocr_service
    result = await ocr_service.process_image(image_path)
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
            "image_height": self.image_height
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
# OCR SERVICE
# =============================================================================

class OCRService:
    """
    Chandra OCR Service - supports GPU and CPU inference.
    
    For GPU: Uses standard InferenceManager (fast)
    For CPU: Uses custom loader with patched inference (slow but works)
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
        
        self._manager = None
        self._model = None
        self._processor = None
        self._model_loaded = False
        self._model_lock = threading.Lock()
        self._use_cpu_mode = False
        self._device = None
        
        # Single worker for CPU mode (memory constraints)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._semaphore = threading.Semaphore(1)
        
        self.max_dimension = settings.OCR_MAX_IMAGE_DIMENSION
        
        self._initialized = True
        logger.info("OCR Service initialized")
    
    # =========================================================================
    # HARDWARE DETECTION
    # =========================================================================
    
    def _detect_hardware(self) -> dict:
        """Detect available hardware."""
        import torch
        
        gpu_available = torch.cuda.is_available()
        gpu_memory = 0
        if gpu_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        try:
            import psutil
            ram_available = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            ram_available = 4.0  # Assume low
        
        return {
            'gpu_available': gpu_available,
            'gpu_memory_gb': gpu_memory,
            'ram_available_gb': ram_available
        }
    
    # =========================================================================
    # MODEL LOADING
    # =========================================================================
    
    def _ensure_model_loaded(self) -> None:
        """Load model with appropriate strategy."""
        if self._model_loaded:
            return
        
        with self._model_lock:
            if self._model_loaded:
                return
            
            hw = self._detect_hardware()
            logger.info(f"Hardware: GPU={hw['gpu_available']}, GPU-MEM={hw['gpu_memory_gb']:.1f}GB, RAM={hw['ram_available_gb']:.1f}GB")
            
            if hw['gpu_available'] and hw['gpu_memory_gb'] >= 4:
                # GPU path - use standard InferenceManager
                self._load_gpu_mode()
            else:
                # CPU path - custom loader
                self._load_cpu_mode()
            
            self._model_loaded = True
    
    def _load_gpu_mode(self) -> None:
        """Load using standard InferenceManager (GPU)."""
        logger.info("Loading Chandra in GPU mode...")
        
        try:
            from chandra.model import InferenceManager
            self._manager = InferenceManager(method="hf")
            self._use_cpu_mode = False
            self._device = "cuda"
            logger.info("GPU mode loaded successfully")
        except Exception as e:
            logger.warning(f"GPU mode failed: {e}, falling back to CPU")
            self._load_cpu_mode()
    
    def _load_cpu_mode(self) -> None:
        """
        Load model for CPU inference.
        
        This requires patching because Chandra's hf.py hardcodes CUDA.
        We load the model manually and use a custom generate function.
        """
        import torch
        
        logger.info("=" * 60)
        logger.info("LOADING CHANDRA IN CPU MODE")
        logger.info("This will take 10-30 minutes and be SLOW per image.")
        logger.info("For better performance, use a GPU with 4GB+ VRAM.")
        logger.info("=" * 60)
        
        # Import the correct model classes from transformers
        # Chandra uses Qwen3VL, so we need these specific classes
        try:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        except ImportError:
            # Fallback for older transformers
            from transformers import AutoModelForVision2Seq, AutoProcessor
            logger.warning("Qwen3VL classes not found, using Auto classes")
            Qwen3VLForConditionalGeneration = AutoModelForVision2Seq
            Qwen3VLProcessor = AutoProcessor
        
        model_checkpoint = "datalab-to/chandra"
        
        logger.info("[1/3] Loading processor...")
        self._processor = Qwen3VLProcessor.from_pretrained(
            model_checkpoint,
            trust_remote_code=True
        )
        
        logger.info("[2/3] Loading model (this takes 10-20 minutes)...")
        logger.info("      Model will be loaded to RAM in chunks.")
        
        # Load with CPU-specific settings
        # Key: device_map must be "cpu" or {"": "cpu"}, NOT "auto"
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_checkpoint,
            device_map={"": "cpu"},  # Force CPU, not "auto"
            torch_dtype=torch.float32,  # Float32 for CPU stability
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self._model = self._model.eval()
        self._model.processor = self._processor
        
        # Set torch threads for CPU performance
        torch.set_num_threads(os.cpu_count() or 4)
        
        self._use_cpu_mode = True
        self._device = "cpu"
        
        logger.info("[3/3] CPU mode loaded!")
        logger.info(f"      Threads: {torch.get_num_threads()}")
    
    # =========================================================================
    # INFERENCE
    # =========================================================================
    
    def _generate_ocr(self, image: Image.Image) -> dict:
        """Run OCR inference."""
        if self._use_cpu_mode:
            return self._generate_cpu(image)
        else:
            return self._generate_via_manager(image)
    
    def _generate_via_manager(self, image: Image.Image) -> dict:
        """Generate using standard InferenceManager (GPU)."""
        from chandra.model.schema import BatchInputItem
        
        batch = [BatchInputItem(image=image, prompt_type="ocr_layout")]
        results = self._manager.generate(batch)
        result = results[0]
        
        return {
            "markdown": result.markdown or "",
            "html": result.html or "",
            "json": {},
            "error": result.error
        }
    
    def _generate_cpu(self, image: Image.Image) -> dict:
        """
        Generate OCR on CPU - custom implementation.
        
        This is a modified version of chandra.model.hf.generate_hf
        that works on CPU instead of hardcoded CUDA.
        """
        import torch
        from chandra.model.util import scale_to_fit
        from chandra.prompts import PROMPT_MAPPING
        from chandra.output import parse_markdown, parse_html
        
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            logger.error("qwen_vl_utils not installed. Run: pip install qwen-vl-utils")
            return {"markdown": "", "html": "", "json": {}, "error": "qwen_vl_utils missing"}
        
        # Scale image to fit model requirements
        scaled_image = scale_to_fit(image)
        
        # Build message (same as chandra's process_batch_element)
        prompt = PROMPT_MAPPING["ocr_layout"]
        content = [
            {"type": "image", "image": scaled_image},
            {"type": "text", "text": prompt}
        ]
        messages = [{"role": "user", "content": content}]
        
        # Process
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, _ = process_vision_info(messages)
        inputs = self._processor(
            text=text,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )
        
        # KEY FIX: Move to CPU, not CUDA
        inputs = inputs.to("cpu")
        
        # Generate with no_grad for memory efficiency
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=4096  # Reduced from 8192 for CPU
            )
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        raw_output = output_text[0] if output_text else ""
        
        # Parse output
        markdown = parse_markdown(raw_output)
        html = parse_html(raw_output)
        
        # Cleanup
        del inputs, generated_ids, generated_ids_trimmed
        gc.collect()
        
        return {
            "markdown": markdown,
            "html": html,
            "json": {},
            "error": None
        }
    
    # =========================================================================
    # PROCESSING
    # =========================================================================
    
    def _process_single_image_sync(
        self,
        image: Image.Image,
        page_number: int = 1
    ) -> OCROutput:
        """Process single image."""
        with self._semaphore:
            start_time = time.time()
            original_size = image.size
            
            try:
                self._ensure_model_loaded()
                
                # Resize for performance
                optimized = image_preprocessor.resize_if_needed(
                    image, 
                    max_dimension=self.max_dimension
                )
                
                logger.info(f"Processing image {page_number} ({optimized.size})...")
                
                output = self._generate_ocr(optimized)
                
                processing_time = int((time.time() - start_time) * 1000)
                
                # Cleanup
                del optimized
                gc.collect()
                
                if output.get("error"):
                    return OCROutput(
                        success=False,
                        error=str(output["error"]),
                        processing_time_ms=processing_time,
                        page_number=page_number,
                        image_width=original_size[0],
                        image_height=original_size[1]
                    )
                
                return OCROutput(
                    markdown=output.get("markdown", ""),
                    html=output.get("html", ""),
                    json_output=output.get("json", {}),
                    processing_time_ms=processing_time,
                    success=True,
                    page_number=page_number,
                    image_width=original_size[0],
                    image_height=original_size[1]
                )
                
            except Exception as e:
                processing_time = int((time.time() - start_time) * 1000)
                logger.error(f"OCR failed: {e}", exc_info=True)
                gc.collect()
                
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
        """Process image from various sources."""
        # Load image
        if isinstance(image_source, bytes):
            image = image_preprocessor.load_image_bytes(image_source)
        elif isinstance(image_source, (str, Path)):
            image = image_preprocessor.load_image(image_source)
        elif isinstance(image_source, Image.Image):
            image = image_source
        else:
            raise ValueError(f"Unsupported image type: {type(image_source)}")
        
        # Preprocess
        image = image_preprocessor.optimize_for_ocr(
            image,
            apply_contrast=True,
            apply_sharpness=True,
            apply_denoise=False
        )
        
        return self._process_single_image_sync(image, page_number)
    
    def process_pdf_sync(self, pdf_path: Union[str, Path]) -> DocumentOCRResult:
        """Process PDF document."""
        start_time = time.time()
        
        try:
            images = image_preprocessor.pdf_to_images(pdf_path)
            
            if not images:
                return DocumentOCRResult(success=False, error="No pages found")
            
            page_results: List[OCROutput] = []
            for i, page_image in enumerate(images, start=1):
                logger.info(f"Processing PDF page {i}/{len(images)}")
                result = self._process_single_image_sync(page_image, i)
                page_results.append(result)
                gc.collect()
            
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
        timeout: float = 600.0
    ) -> OCROutput:
        """Async image processing with timeout."""
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
        timeout: float = 1800.0
    ) -> DocumentOCRResult:
        """Async PDF processing."""
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
        """Process any document type."""
        file_type = file_type.lower().strip('.')
        path = Path(file_path)
        
        if not path.exists():
            return DocumentOCRResult(success=False, error=f"File not found: {path}")
        
        if file_type == 'pdf':
            return await self.process_pdf(path)
        elif file_type in ('png', 'jpg', 'jpeg'):
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
            return DocumentOCRResult(success=False, error=f"Unsupported: {file_type}")
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _combine_markdown(self, pages: List[OCROutput]) -> str:
        sections = []
        for page in pages:
            if page.markdown:
                if len(pages) > 1:
                    sections.append(f"## Page {page.page_number}\n\n{page.markdown}")
                else:
                    sections.append(page.markdown)
        return "\n\n---\n\n".join(sections)
    
    def _combine_html(self, pages: List[OCROutput]) -> str:
        sections = []
        for page in pages:
            if page.html:
                if len(pages) > 1:
                    sections.append(f'<section data-page="{page.page_number}">\n{page.html}\n</section>')
                else:
                    sections.append(page.html)
        return "\n<hr>\n".join(sections)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "model_loaded": self._model_loaded,
            "mode": "cpu" if self._use_cpu_mode else "gpu",
            "device": self._device
        }
    
    def preload_model(self) -> None:
        self._ensure_model_loaded()
    
    @property
    def is_model_loaded(self) -> bool:
        return self._model_loaded
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=False)
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._model_loaded = False
        gc.collect()
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
