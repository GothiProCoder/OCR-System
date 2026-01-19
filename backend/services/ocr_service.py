"""
OCR Service - Azure Document Intelligence Implementation
=========================================================
Document parsing using Azure AI Document Intelligence (formerly Form Recognizer).

Features:
- Cloud-based OCR with high accuracy
- Markdown output for seamless LLM integration
- Table, paragraph, and word-level extraction
- Bounding box data for frontend visualization
- Built-in preprocessing (deskew, compression)
- PDF and image support

Azure Model Options:
- prebuilt-layout: Full extraction (text, tables, figures, structure)
- prebuilt-read: Fast text-only extraction

Requires:
- azure-ai-documentintelligence package
- Azure Document Intelligence resource credentials in .env
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
    processed_image_bytes: Optional[bytes] = None  # Preprocessed image for frontend
    page_width_inches: float = 0.0  # Azure page dimensions
    page_height_inches: float = 0.0
    
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
            "layout_boxes": self.layout_boxes,
            "page_width_inches": self.page_width_inches,
            "page_height_inches": self.page_height_inches,
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
    combined_layout_boxes: List[Dict[str, Any]] = field(default_factory=list)  # All pages' boxes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages": [p.to_dict() for p in self.pages],
            "total_pages": self.total_pages,
            "total_processing_time_ms": self.total_processing_time_ms,
            "success": self.success,
            "error": self.error,
            "combined_markdown": self.combined_markdown,
            "combined_html": self.combined_html,
            "combined_layout_boxes": self.combined_layout_boxes,
        }


# =============================================================================
# OCR SERVICE - Azure Document Intelligence Implementation
# =============================================================================

class OCRService:
    """
    Azure Document Intelligence OCR Service.
    
    Features:
    - Cloud-based processing (no local GPU needed)
    - Markdown output with tables
    - Bounding boxes for UI overlay
    - Built-in preprocessing pipeline
    
    Azure Requirements:
    - AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT in .env
    - AZURE_DOCUMENT_INTELLIGENCE_KEY in .env
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
        
        self._client = None
        self._client_lock = threading.Lock()
        
        # Configuration from settings
        self._endpoint = getattr(settings, 'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT', '')
        self._api_key = getattr(settings, 'AZURE_DOCUMENT_INTELLIGENCE_KEY', '')
        self._model_id = getattr(settings, 'AZURE_DOCUMENT_INTELLIGENCE_MODEL', 'prebuilt-layout')
        
        # Preprocessing settings
        self._apply_deskew = getattr(settings, 'PREPROCESSING_APPLY_DESKEW', True)
        self._apply_binarize = getattr(settings, 'PREPROCESSING_APPLY_BINARIZE', False)
        self._target_size_mb = getattr(settings, 'PREPROCESSING_TARGET_SIZE_MB', 2.0)
        self.max_dimension = getattr(settings, 'OCR_MAX_IMAGE_DIMENSION', 2000)
        
        # Single worker for sequential processing
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._semaphore = threading.Semaphore(1)
        
        self._initialized = True
        logger.info(f"OCR Service initialized (Azure Document Intelligence mode, model: {self._model_id})")
    
    # =========================================================================
    # CLIENT MANAGEMENT
    # =========================================================================
    
    def _ensure_client_initialized(self) -> None:
        """Lazy-load Azure Document Intelligence client."""
        if self._client is not None:
            return
        
        with self._client_lock:
            if self._client is not None:
                return
            
            if not self._endpoint or not self._api_key:
                raise ValueError(
                    "Azure Document Intelligence credentials not configured. "
                    "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
                    "AZURE_DOCUMENT_INTELLIGENCE_KEY in your .env file."
                )
            
            logger.info("=" * 60)
            logger.info("INITIALIZING Azure Document Intelligence Client")
            logger.info(f"Endpoint: {self._endpoint[:50]}...")
            logger.info(f"Model: {self._model_id}")
            logger.info("=" * 60)
            
            try:
                from azure.core.credentials import AzureKeyCredential
                from azure.ai.documentintelligence import DocumentIntelligenceClient
                
                self._client = DocumentIntelligenceClient(
                    endpoint=self._endpoint,
                    credential=AzureKeyCredential(self._api_key)
                )
                
                logger.info("Azure Document Intelligence client initialized successfully")
                
            except ImportError as e:
                logger.error(f"Azure SDK not installed: {e}")
                raise RuntimeError(
                    "Azure Document Intelligence SDK not installed. "
                    "Run: pip install azure-ai-documentintelligence"
                )
            except Exception as e:
                logger.error(f"Failed to initialize Azure client: {e}")
                raise
    
    # =========================================================================
    # CORE AZURE PROCESSING
    # =========================================================================
    
    def _analyze_with_azure(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Call Azure Document Intelligence API.
        
        Args:
            file_bytes: Document bytes (JPEG for images, PDF for PDFs)
            
        Returns:
            Parsed AnalyzeResult as dictionary
        """
        self._ensure_client_initialized()
        
        try:
            from azure.ai.documentintelligence.models import DocumentContentFormat
            
            logger.info(f"Sending {len(file_bytes)/1024:.1f}KB to Azure...")
            
            # Analyze with Markdown output format
            poller = self._client.begin_analyze_document(
                self._model_id,
                body=file_bytes,
                output_content_format=DocumentContentFormat.MARKDOWN,
                content_type="application/octet-stream"
            )
            
            result = poller.result()
            
            logger.info(f"Azure analysis complete: {len(result.pages)} pages")
            
            return result
            
        except Exception as e:
            logger.error(f"Azure API error: {e}")
            raise
    
    def _extract_layout_boxes(self, result, page_number: int = 1) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes from Azure result for frontend visualization.
        
        Extracts:
        - Words with confidence and polygons
        - Lines with text
        - Tables with cell positions
        - Paragraphs with roles
        """
        layout_boxes = []
        
        try:
            # Get the specific page
            if not result.pages or page_number > len(result.pages):
                return layout_boxes
            
            page = result.pages[page_number - 1]
            
            # Helper function to safely extract polygon coordinates
            def extract_polygon(polygon) -> List[float]:
                """Extract polygon coordinates - Azure returns flat list of floats."""
                if not polygon:
                    return []
                try:
                    # Azure already returns [x1, y1, x2, y2, ...] as floats
                    coords = list(polygon)
                    
                    # Validate: should have even number of coordinates (x,y pairs)
                    if len(coords) % 2 != 0:
                        logger.warning(f"Polygon has odd number of coordinates: {len(coords)}")
                    
                    return coords
                except Exception as e:
                    logger.error(f"Error extracting polygon: {e}. Type: {type(polygon)}")
                    return []
            
            # Extract words
            if page.words:
                logger.info(f"Page {page_number}: Extracting {len(page.words)} words")
                # Log first word polygon as sample
                if len(page.words) > 0:
                    first_word = page.words[0]
                    logger.debug(f"Sample word polygon - Type: {type(first_word.polygon)}, Length: {len(list(first_word.polygon)) if first_word.polygon else 0}")
                
                for word in page.words:
                    polygon_coords = extract_polygon(word.polygon)
                    layout_boxes.append({
                        "type": "word",
                        "content": word.content,
                        "confidence": word.confidence,
                        "polygon": polygon_coords,
                        "page_number": page_number
                    })
            
            # Extract lines
            if page.lines:
                for line in page.lines:
                    layout_boxes.append({
                        "type": "line",
                        "content": line.content,
                        "polygon": extract_polygon(line.polygon),
                        "page_number": page_number
                    })
            
            # Extract selection marks (checkboxes)
            if page.selection_marks:
                for mark in page.selection_marks:
                    layout_boxes.append({
                        "type": "selection_mark",
                        "state": mark.state,  # "selected" or "unselected"
                        "confidence": mark.confidence,
                        "polygon": extract_polygon(mark.polygon),
                        "page_number": page_number
                    })
            
            # Extract tables
            if result.tables:
                for table_idx, table in enumerate(result.tables):
                    # Check if table is on this page
                    if table.bounding_regions:
                        for region in table.bounding_regions:
                            if region.page_number == page_number:
                                layout_boxes.append({
                                    "type": "table",
                                    "table_index": table_idx,
                                    "row_count": table.row_count,
                                    "column_count": table.column_count,
                                    "polygon": extract_polygon(region.polygon),
                                    "page_number": page_number
                                })
                    
                    # Extract cells
                    for cell in table.cells:
                        if cell.bounding_regions:
                            for region in cell.bounding_regions:
                                if region.page_number == page_number:
                                    layout_boxes.append({
                                        "type": "table_cell",
                                        "content": cell.content,
                                        "row_index": cell.row_index,
                                        "column_index": cell.column_index,
                                        "polygon": extract_polygon(region.polygon),
                                        "page_number": page_number
                                    })
            
            # Extract paragraphs with roles (titles, headers, etc.)
            if result.paragraphs:
                for para in result.paragraphs:
                    if para.bounding_regions:
                        for region in para.bounding_regions:
                            if region.page_number == page_number:
                                layout_boxes.append({
                                    "type": "paragraph",
                                    "content": para.content[:100] + "..." if len(para.content) > 100 else para.content,
                                    "role": para.role if para.role else "text",
                                    "polygon": extract_polygon(region.polygon),
                                    "page_number": page_number
                                })
                                break  # Only first region per paragraph
            
        except Exception as e:
            logger.error(f"Error extracting layout boxes: {e}", exc_info=True)
        
        logger.info(f"Page {page_number}: Extracted {len(layout_boxes)} total layout boxes")
        if layout_boxes:
            logger.debug(f"Sample box: {layout_boxes[0]}")
        
        return layout_boxes
    
    def _generate_html_from_markdown(self, markdown_text: str) -> str:
        """Convert markdown to simple HTML."""
        try:
            # Simple markdown to HTML conversion
            # Replace headers
            html = markdown_text
            
            # Tables are already HTML in Azure's markdown output
            # Just wrap in basic structure
            html = f"<div class='ocr-content'>\n{html}\n</div>"
            
            return html
        except Exception as e:
            logger.warning(f"HTML generation failed: {e}")
            return f"<pre>{markdown_text}</pre>"
    
    # =========================================================================
    # IMAGE PROCESSING
    # =========================================================================
    
    def _process_single_image_sync(
        self,
        image: Image.Image,
        page_number: int = 1
    ) -> OCROutput:
        """Process a single image synchronously with Azure."""
        with self._semaphore:
            start_time = time.time()
            original_size = image.size
            
            try:
                # Preprocess image for Azure
                logger.info(f"Preprocessing image page {page_number} ({image.size})...")
                
                preprocessed_bytes = image_preprocessor.preprocess_for_azure(
                    image,
                    apply_deskew=self._apply_deskew,
                    apply_binarize=self._apply_binarize,
                    target_size_mb=self._target_size_mb
                )
                
                # Send to Azure
                result = self._analyze_with_azure(preprocessed_bytes)
                
                processing_time = int((time.time() - start_time) * 1000)
                
                # Extract markdown (Azure returns it in result.content with Markdown format)
                markdown_text = result.content if result.content else ""
                
                # Extract layout boxes for this page
                layout_boxes = self._extract_layout_boxes(result, page_number=page_number)
                
                # Generate HTML
                html_content = self._generate_html_from_markdown(markdown_text)
                
                # Extract page dimensions from Azure result
                page_width_inches = 0.0
                page_height_inches = 0.0
                if result.pages and len(result.pages) > 0:
                    page = result.pages[0]
                    page_width_inches = float(page.width) if page.width else 0.0
                    page_height_inches = float(page.height) if page.height else 0.0
                
                # Build JSON output summary
                json_output = {
                    "page_count": len(result.pages) if result.pages else 1,
                    "words_count": sum(len(p.words) if p.words else 0 for p in result.pages) if result.pages else 0,
                    "tables_count": len(result.tables) if result.tables else 0,
                    "paragraphs_count": len(result.paragraphs) if result.paragraphs else 0,
                }
                
                return OCROutput(
                    markdown=markdown_text,
                    html=html_content,
                    json_output=json_output,
                    processing_time_ms=processing_time,
                    success=True,
                    page_number=page_number,
                    image_width=original_size[0],
                    image_height=original_size[1],
                    layout_boxes=layout_boxes,
                    processed_image_bytes=preprocessed_bytes,  # Store for saving later
                    page_width_inches=page_width_inches,
                    page_height_inches=page_height_inches,
                )
                
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
        
        return self._process_single_image_sync(image, page_number)
    
    # =========================================================================
    # PDF PROCESSING
    # =========================================================================
    
    def process_pdf_sync(self, pdf_path: Union[str, Path]) -> DocumentOCRResult:
        """
        Process a PDF document synchronously.
        
        Azure Document Intelligence can process PDFs directly.
        
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
            
            self._ensure_client_initialized()
            
            logger.info(f"Processing PDF: {pdf_path.name}")
            
            # Read PDF file
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            # Check size and compress if needed
            if len(pdf_bytes) > self._target_size_mb * 1024 * 1024:
                logger.warning(f"PDF size {len(pdf_bytes)/1024/1024:.1f}MB exceeds target, processing as images")
                return self.process_pdf_as_images_sync(pdf_path)
            
            # Send to Azure
            result = self._analyze_with_azure(pdf_bytes)
            
            page_results: List[OCROutput] = []
            
            # Process each page
            num_pages = len(result.pages) if result.pages else 1
            
            for page_num in range(1, num_pages + 1):
                page = result.pages[page_num - 1] if result.pages else None
                
                # Get page dimensions
                page_width = page.width if page else 0
                page_height = page.height if page else 0
                
                # Extract layout boxes for this page
                layout_boxes = self._extract_layout_boxes(result, page_number=page_num)
                
                # For multi-page PDFs, we need to split the content
                # Azure returns combined markdown, so we use page markers or just assign to pages
                page_markdown = f"<!-- Page {page_num} -->\n"
                if num_pages == 1:
                    page_markdown = result.content if result.content else ""
                
                page_results.append(OCROutput(
                    markdown=page_markdown,
                    html=self._generate_html_from_markdown(page_markdown),
                    json_output={"page": page_num},
                    processing_time_ms=0,  # Will sum at end
                    success=True,
                    page_number=page_num,
                    image_width=int(page_width) if page_width else 0,
                    image_height=int(page_height) if page_height else 0,
                    layout_boxes=layout_boxes
                ))
            
            # Combined markdown from Azure
            combined_md = result.content if result.content else ""
            combined_html = self._generate_html_from_markdown(combined_md)
            
            total_time = int((time.time() - start_time) * 1000)
            
            return DocumentOCRResult(
                pages=page_results,
                total_pages=len(page_results),
                total_processing_time_ms=total_time,
                success=True,
                error=None,
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
        Process PDF by converting to images first.
        
        Used when PDF is too large for direct Azure processing.
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
            
            # Combine layout boxes from all pages
            combined_boxes = []
            for page in page_results:
                combined_boxes.extend(page.layout_boxes)
            
            total_time = int((time.time() - start_time) * 1000)
            all_success = all(p.success for p in page_results)
            
            return DocumentOCRResult(
                pages=page_results,
                total_pages=len(page_results),
                total_processing_time_ms=total_time,
                success=all_success,
                error=None if all_success else "Some pages failed",
                combined_markdown=combined_md,
                combined_html=combined_html,
                combined_layout_boxes=combined_boxes,
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
        timeout: float = 120.0  # 2 min for Azure API
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
        timeout: float = 600.0  # 10 min for multi-page PDFs
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
                combined_html=result.html,
                combined_layout_boxes=result.layout_boxes,
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
            "client_initialized": self._client is not None,
            "model_id": self._model_id,
            "endpoint_configured": bool(self._endpoint),
            "api_key_configured": bool(self._api_key),
            "apply_deskew": self._apply_deskew,
            "apply_binarize": self._apply_binarize,
            "target_size_mb": self._target_size_mb,
            "max_dimension": self.max_dimension,
            "engine": "Azure Document Intelligence"
        }
    
    def preload_model(self) -> None:
        """Initialize Azure client at startup."""
        self._ensure_client_initialized()
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if client is initialized."""
        return self._client is not None
    
    def cleanup(self) -> None:
        """Full cleanup."""
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except:
                    pass
                self._client = None
        
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
    """Initialize Azure client at startup."""
    try:
        ocr_service.preload_model()
    except Exception as e:
        logger.warning(f"OCR preload failed: {e}")


async def get_ocr_status() -> Dict[str, Any]:
    """Get service status."""
    return ocr_service.get_status()
