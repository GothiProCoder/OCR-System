"""
OCR Service - MEMORY OPTIMIZED FOR 8GB RAM
===========================================
Implements ALL 11 memory optimization fixes:

#1: Model unloads after each request (load-on-demand)
#2: torch.float16 instead of float32 (50% memory reduction)
#3: device_map='auto' with max_memory + offload_folder
#4: max_memory={'cpu': '4GB'} to limit RAM usage
#5: max_new_tokens=512 with cache management
#6: Complete tensor cleanup (ALL intermediate variables)
#7: Memory monitoring with forced cleanup at 80% RAM
#8: low_cpu_mem_usage=True + max_memory combined
#9: Immediate deletion of image copies
#10: Enhanced cleanup with PyTorch-specific commands
#11: BONUS: 4-bit quantization option for extreme constraints

Target: Run Chandra OCR on 8GB RAM without crashing
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
# OCR SERVICE - MEMORY OPTIMIZED
# =============================================================================

class OCRService:
    """
    Memory-optimized Chandra OCR Service for 8GB RAM systems.
    
    Key optimizations:
    - Load model on demand, unload after each request (#1)
    - Float16 precision (#2)
    - Disk offloading with 4GB max RAM (#3, #4)
    - Reduced token generation (#5)
    - Aggressive memory cleanup (#6, #10)
    - Memory monitoring (#7)
    - Optional 4-bit quantization (#11)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Configuration
    USE_4BIT_QUANTIZATION = False  # Set True for extreme memory constraints
    MAX_CPU_MEMORY = "4GB"  # FIX #4: Limit RAM usage
    MAX_NEW_TOKENS = 512  # FIX #5: Reduced from 4096
    MEMORY_THRESHOLD = 0.80  # FIX #7: Force cleanup at 80% RAM
    UNLOAD_AFTER_REQUEST = True  # FIX #1: Unload model after each request
    
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
        
        # Single worker to prevent memory accumulation
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._semaphore = threading.Semaphore(1)
        
        self.max_dimension = settings.OCR_MAX_IMAGE_DIMENSION
        
        # Offload directory for disk offloading
        self._offload_dir = Path(settings.PROJECT_ROOT) / ".model_offload"
        self._offload_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
        logger.info("OCR Service initialized (memory-optimized mode)")
    
    # =========================================================================
    # FIX #7 & #10: MEMORY MONITORING & ENHANCED CLEANUP
    # =========================================================================
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage as percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return 0.5  # Assume 50% if psutil not available
    
    def _force_cleanup(self) -> None:
        """
        FIX #10: Enhanced cleanup with PyTorch-specific commands.
        Maximizes memory reclamation.
        """
        import torch
        
        # Triple gc.collect for thorough cleanup
        gc.collect()
        gc.collect()
        gc.collect()
        
        # PyTorch-specific cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear any cached tensors
        if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
            try:
                torch._C._cuda_clearCublasWorkspaces()
            except:
                pass
        
        # Linux-specific: trim malloc
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass  # Not on Linux or libc not available
        
        gc.collect()
    
    def _check_memory_and_cleanup(self) -> None:
        """FIX #7: Check memory and force cleanup if needed."""
        mem_usage = self._get_memory_usage()
        if mem_usage > self.MEMORY_THRESHOLD:
            logger.warning(f"Memory at {mem_usage*100:.1f}%, forcing cleanup...")
            self._force_cleanup()
            
            # If still high, unload model
            if self._get_memory_usage() > self.MEMORY_THRESHOLD:
                logger.warning("Memory still high, unloading model...")
                self._unload_model()
    
    # =========================================================================
    # FIX #1: MODEL UNLOADING
    # =========================================================================
    
    def _unload_model(self) -> None:
        """FIX #1: Unload model to free memory."""
        with self._model_lock:
            if self._model is not None:
                logger.info("Unloading model to free memory...")
                del self._model
                self._model = None
            if self._processor is not None:
                del self._processor
                self._processor = None
            if self._manager is not None:
                del self._manager
                self._manager = None
            
            self._model_loaded = False
            self._force_cleanup()
            logger.info("Model unloaded, memory freed")
    
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
            ram_total = psutil.virtual_memory().total / (1024**3)
            ram_available = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            ram_total = 8.0
            ram_available = 4.0
        
        return {
            'gpu_available': gpu_available,
            'gpu_memory_gb': gpu_memory,
            'ram_total_gb': ram_total,
            'ram_available_gb': ram_available
        }
    
    # =========================================================================
    # MODEL LOADING - MEMORY OPTIMIZED
    # =========================================================================
    
    def _ensure_model_loaded(self) -> None:
        """Load model with memory optimizations."""
        if self._model_loaded:
            return
        
        with self._model_lock:
            if self._model_loaded:
                return
            
            # Check memory before loading
            self._check_memory_and_cleanup()
            
            hw = self._detect_hardware()
            logger.info(f"Hardware: GPU={hw['gpu_available']}, RAM={hw['ram_available_gb']:.1f}GB available")
            
            if hw['gpu_available'] and hw['gpu_memory_gb'] >= 4:
                self._load_gpu_mode()
            else:
                self._load_cpu_mode_optimized()
            
            self._model_loaded = True
    
    def _load_gpu_mode(self) -> None:
        """Load using standard InferenceManager (GPU)."""
        logger.info("Loading Chandra in GPU mode...")
        
        try:
            from chandra.model import InferenceManager
            self._manager = InferenceManager(method="hf")
            self._use_cpu_mode = False
            self._device = "cuda"
            logger.info("GPU mode loaded")
        except Exception as e:
            logger.warning(f"GPU mode failed: {e}, falling back to CPU")
            self._load_cpu_mode_optimized()
    
    def _load_cpu_mode_optimized(self) -> None:
        """
        FIX #2, #3, #4, #8, #11: Memory-optimized CPU loading.
        
        - float16 instead of float32 (#2)
        - device_map='auto' with max_memory + offload_folder (#3)
        - max_memory={'cpu': '4GB'} (#4)
        - low_cpu_mem_usage=True (#8)
        - Optional 4-bit quantization (#11)
        """
        import torch
        
        logger.info("=" * 60)
        logger.info("LOADING CHANDRA - MEMORY OPTIMIZED MODE")
        logger.info(f"Max CPU Memory: {self.MAX_CPU_MEMORY}")
        logger.info(f"4-bit Quantization: {self.USE_4BIT_QUANTIZATION}")
        logger.info("This will take 10-30 minutes. Please wait...")
        logger.info("=" * 60)
        
        # Import correct model classes
        try:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        except ImportError:
            from transformers import AutoModelForVision2Seq as Qwen3VLForConditionalGeneration
            from transformers import AutoProcessor as Qwen3VLProcessor
            logger.warning("Using fallback model classes")
        
        model_checkpoint = "datalab-to/chandra"
        
        # Load processor first (lightweight)
        logger.info("[1/2] Loading processor...")
        self._processor = Qwen3VLProcessor.from_pretrained(
            model_checkpoint,
            trust_remote_code=True
        )
        
        # FIX #11: Optional 4-bit quantization
        quantization_config = None
        if self.USE_4BIT_QUANTIZATION:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                logger.info("Using 4-bit quantization (87% memory reduction)")
            except ImportError:
                logger.warning("bitsandbytes not installed, skipping 4-bit quantization")
        
        # Load model with ALL memory optimizations
        logger.info("[2/2] Loading model with memory optimizations...")
        
        # FIX #2: float16 instead of float32
        # FIX #3: device_map='auto' with offload
        # FIX #4: max_memory to limit RAM
        # FIX #8: low_cpu_mem_usage=True
        load_kwargs = {
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,  # FIX #8
            'torch_dtype': torch.float16,  # FIX #2: NOT float32
            'device_map': 'auto',  # FIX #3
            'max_memory': {'cpu': self.MAX_CPU_MEMORY},  # FIX #4
            'offload_folder': str(self._offload_dir),  # FIX #3
            'offload_state_dict': True,  # FIX #3
        }
        
        if quantization_config:
            load_kwargs['quantization_config'] = quantization_config
        
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_checkpoint,
            **load_kwargs
        )
        self._model = self._model.eval()
        self._model.processor = self._processor
        
        # Set torch threads for CPU
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        
        self._use_cpu_mode = True
        self._device = "cpu"
        
        logger.info(f"Model loaded (threads={torch.get_num_threads()})")
        logger.info(f"Current memory: {self._get_memory_usage()*100:.1f}%")
    
    # =========================================================================
    # INFERENCE - MEMORY OPTIMIZED
    # =========================================================================
    
    def _generate_ocr(self, image: Image.Image) -> dict:
        """Run OCR inference with memory management."""
        if self._use_cpu_mode:
            return self._generate_cpu_optimized(image)
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
            "error": result.error
        }
    
    def _generate_cpu_optimized(self, image: Image.Image) -> dict:
        """
        FIX #5, #6: Memory-optimized CPU generation.
        
        - max_new_tokens=512 (reduced from 4096) (#5)
        - use_cache=True with do_sample=False (#5)
        - Complete tensor cleanup (#6)
        """
        import torch
        from chandra.model.util import scale_to_fit
        from chandra.prompts import PROMPT_MAPPING
        from chandra.output import parse_markdown, parse_html
        
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            return {"markdown": "", "html": "", "error": "qwen_vl_utils missing"}
        
        # Scale image
        scaled_image = scale_to_fit(image)
        
        # Build message
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
        
        # Move to appropriate device
        inputs = inputs.to(self._device)
        
        # FIX #5: Optimized generation with reduced tokens
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,  # FIX #5: Reduced from 4096
                use_cache=True,  # FIX #5: Enable KV cache
                do_sample=False,  # FIX #5: Deterministic generation
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
        
        # FIX #6: Complete tensor cleanup - ALL intermediate variables
        del inputs
        del generated_ids
        del generated_ids_trimmed
        del image_inputs
        del scaled_image
        del messages
        del text
        if 'output_text' in locals():
            del output_text
        if 'raw_output' in locals():
            del raw_output
        
        self._force_cleanup()
        
        return {
            "markdown": markdown,
            "html": html,
            "error": None
        }
    
    # =========================================================================
    # PROCESSING - MEMORY OPTIMIZED
    # =========================================================================
    
    def _process_single_image_sync(
        self,
        image: Image.Image,
        page_number: int = 1
    ) -> OCROutput:
        """Process single image with memory management."""
        with self._semaphore:
            # FIX #7: Check memory before processing
            self._check_memory_and_cleanup()
            
            start_time = time.time()
            original_size = image.size
            
            try:
                self._ensure_model_loaded()
                
                # Resize for performance
                optimized = image_preprocessor.resize_if_needed(
                    image, 
                    max_dimension=self.max_dimension
                )
                
                # FIX #9: Delete original image copy immediately
                if image is not optimized:
                    del image
                
                logger.info(f"Processing image {page_number} ({optimized.size})...")
                
                output = self._generate_ocr(optimized)
                
                processing_time = int((time.time() - start_time) * 1000)
                
                # FIX #9: Cleanup optimized image
                del optimized
                
                # FIX #1: Unload model after request if configured
                if self.UNLOAD_AFTER_REQUEST:
                    self._unload_model()
                else:
                    self._force_cleanup()
                
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
                    processing_time_ms=processing_time,
                    success=True,
                    page_number=page_number,
                    image_width=original_size[0],
                    image_height=original_size[1]
                )
                
            except Exception as e:
                processing_time = int((time.time() - start_time) * 1000)
                logger.error(f"OCR failed: {e}", exc_info=True)
                
                # Always cleanup on error
                self._unload_model()
                
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
        
        # FIX #9: Delete source reference if it was loaded
        if isinstance(image_source, (str, Path, bytes)):
            del image_source
        
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
                
                # FIX #9: Delete page image immediately
                del page_image
                
                # FIX #7: Check memory between pages
                self._check_memory_and_cleanup()
            
            # FIX #9: Delete images list
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
            self._unload_model()
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
        """Async image processing."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.process_image_sync, image_source, page_number),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self._unload_model()
            return OCROutput(success=False, error=f"Timed out after {timeout}s")
    
    async def process_pdf(
        self,
        pdf_path: Union[str, Path],
        timeout: float = 3600.0  # 1 hour for multi-page PDFs
    ) -> DocumentOCRResult:
        """Async PDF processing."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.process_pdf_sync, pdf_path),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self._unload_model()
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
            "device": self._device,
            "memory_usage": f"{self._get_memory_usage()*100:.1f}%",
            "max_memory": self.MAX_CPU_MEMORY,
            "unload_after_request": self.UNLOAD_AFTER_REQUEST,
            "4bit_quantization": self.USE_4BIT_QUANTIZATION
        }
    
    def preload_model(self) -> None:
        """Preload model (not recommended for memory-constrained systems)."""
        self._ensure_model_loaded()
    
    @property
    def is_model_loaded(self) -> bool:
        return self._model_loaded
    
    def cleanup(self) -> None:
        """Full cleanup."""
        self._unload_model()
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
    """Preload model at startup (not recommended for 8GB systems)."""
    try:
        ocr_service.preload_model()
    except Exception as e:
        logger.warning(f"OCR preload failed: {e}")


async def get_ocr_status() -> Dict[str, Any]:
    """Get service status."""
    return ocr_service.get_status()
