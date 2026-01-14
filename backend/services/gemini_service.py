"""
Gemini Service - LLM Key-Value Extraction
==========================================
Production-grade integration with Google Gen AI SDK for extracting
structured key-value pairs from OCR text.

Features:
- Structured JSON output using Pydantic response schemas
- Automatic field type detection
- Confidence scoring for extracted values
- Error handling with APIError
- Async support for high throughput
- Context manager for resource management
- Token counting for cost optimization

Based on Google Gen AI Python SDK:
    https://googleapis.github.io/python-genai/
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from enum import Enum
import time
import logging
import asyncio
import threading
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# =============================================================================

class FieldTypeEnum(str, Enum):
    """Types of extracted fields"""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    EMAIL = "email"
    PHONE = "phone"
    CHECKBOX = "checkbox"
    TABLE = "table"
    SIGNATURE = "signature"
    ADDRESS = "address"
    NAME = "name"
    CURRENCY = "currency"
    UNKNOWN = "unknown"


class ExtractedFieldSchema(BaseModel):
    """Schema for a single extracted field - used for LLM response parsing"""
    field_key: str = Field(..., description="The label or name of the field (e.g., 'Full Name', 'Date of Birth')")
    field_value: str = Field(..., description="The extracted value for this field")
    field_type: FieldTypeEnum = Field(FieldTypeEnum.TEXT, description="The detected type of the field")
    confidence: float = Field(0.85, ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")


class ExtractionResponseSchema(BaseModel):
    """Schema for the complete LLM extraction response"""
    form_type: str = Field("Unknown", description="The type of form detected (e.g., 'Invoice', 'Application Form', 'Medical Record')")
    language: str = Field("en", description="Language code of the document (e.g., 'en', 'hi', 'es')")
    fields: List[ExtractedFieldSchema] = Field(default_factory=list, description="List of extracted key-value pairs")


# =============================================================================
# DATA CLASSES FOR INTERNAL USE
# =============================================================================

@dataclass
class GeminiExtractionResult:
    """Result from Gemini extraction."""
    fields: List[Dict[str, Any]] = field(default_factory=list)
    form_type: str = "Unknown"
    language: str = "en"
    raw_response: Optional[str] = None
    processing_time_ms: int = 0
    token_count: int = 0
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": self.fields,
            "form_type": self.form_type,
            "language": self.language,
            "raw_response": self.raw_response,
            "processing_time_ms": self.processing_time_ms,
            "token_count": self.token_count,
            "success": self.success,
            "error": self.error
        }


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

DEFAULT_SYSTEM_INSTRUCTION = """You are an expert document analysis AI specializing in extracting structured information from forms, invoices, applications, and other documents.

Your task is to analyze the OCR text provided and extract ALL key-value pairs present in the document.

Guidelines:
1. Extract EVERY field you can identify, including headers, labels, and their corresponding values
2. For handwritten or unclear text, provide your best interpretation and lower confidence
3. Detect the field type (text, number, date, email, phone, address, name, currency, etc.)
4. Assign confidence scores:
   - 0.95-1.0: Very clear, printed text that's easy to read
   - 0.80-0.94: Clear but may have minor ambiguity
   - 0.60-0.79: Readable but some uncertainty
   - 0.40-0.59: Partially legible, best guess
   - Below 0.40: Highly uncertain

5. Identify the form type (Invoice, Application, Medical Form, Survey, etc.)
6. Detect the document language

Be thorough - extract ALL visible fields, not just the main ones."""


# =============================================================================
# GEMINI SERVICE
# =============================================================================

class GeminiService:
    """
    Production-grade Gemini LLM Service for key-value extraction.
    
    Features:
    - Structured JSON output using response schemas
    - Automatic retry with exponential backoff
    - Token counting for cost optimization
    - Async support
    - Context manager for resource cleanup
    - LangGraph node integration
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
        self._model_name = getattr(settings, 'GEMINI_MODEL', 'gemini-2.5-flash-lite')
        self._api_key = getattr(settings, 'GEMINI_API_KEY', '')
        
        # Configuration
        self._temperature = 0.1  # Low for deterministic extraction
        self._max_output_tokens = 8192  # Allow for large form extractions
        self._top_p = 0.95
        
        # System instruction
        self._system_instruction = DEFAULT_SYSTEM_INSTRUCTION
        
        self._initialized = True
        logger.info(f"Gemini Service initialized (model: {self._model_name})")
    
    # =========================================================================
    # CLIENT MANAGEMENT
    # =========================================================================
    
    def _ensure_client(self) -> None:
        """Lazy-load the Gemini client."""
        if self._client is not None:
            return
        
        with self._client_lock:
            if self._client is not None:
                return
            
            if not self._api_key:
                raise ValueError(
                    "GEMINI_API_KEY not configured. "
                    "Set GEMINI_API_KEY in your .env file."
                )
            
            try:
                from google import genai
                
                # Initialize client with API key
                self._client = genai.Client(api_key=self._api_key)
                
                logger.info("Gemini client initialized successfully")
                
            except ImportError as e:
                logger.error(f"google-genai not installed: {e}")
                raise RuntimeError(
                    "google-genai package not installed. "
                    "Run: pip install google-genai"
                )
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                raise
    
    def close(self) -> None:
        """Close the client and free resources."""
        with self._client_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except:
                    pass
                self._client = None
                logger.info("Gemini client closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    # =========================================================================
    # CORE EXTRACTION
    # =========================================================================
    
    def extract_from_text(
        self,
        ocr_text: str,
        custom_prompt: Optional[str] = None,
        form_template: Optional[Dict[str, Any]] = None,
        max_retries: int = 2
    ) -> GeminiExtractionResult:
        """
        Extract key-value pairs from OCR text synchronously.
        
        Includes automatic retry logic for:
        - JSON parsing failures
        - Schema validation errors
        - API errors (with exponential backoff)
        
        Args:
            ocr_text: The OCR markdown/text output
            custom_prompt: Optional custom extraction prompt
            form_template: Optional template with expected fields
            max_retries: Maximum number of retry attempts (default: 2)
            
        Returns:
            GeminiExtractionResult with extracted fields
        """
        start_time = time.time()
        last_error = None
        last_raw_response = None
        
        for attempt in range(max_retries + 1):
            try:
                self._ensure_client()
                
                from google.genai import types
                
                # Build the extraction prompt
                # On retry, add error context to help Gemini correct
                if attempt == 0:
                    prompt = self._build_prompt(ocr_text, custom_prompt, form_template)
                else:
                    prompt = self._build_retry_prompt(
                        ocr_text, 
                        custom_prompt, 
                        form_template,
                        last_error,
                        last_raw_response,
                        attempt
                    )
                    logger.info(f"Retry attempt {attempt}/{max_retries} for extraction")
                
                # Configure generation with JSON response schema
                config = types.GenerateContentConfig(
                    system_instruction=self._system_instruction,
                    temperature=self._temperature,
                    max_output_tokens=self._max_output_tokens,
                    top_p=self._top_p,
                    response_mime_type="application/json",
                    response_schema=ExtractionResponseSchema
                )
                
                # Generate content
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=config
                )
                
                # Try to parse the response
                result = self._parse_response(response)
                
                if result.success:
                    result.processing_time_ms = int((time.time() - start_time) * 1000)
                    if attempt > 0:
                        logger.info(f"Extraction succeeded on retry attempt {attempt}")
                    return result
                else:
                    # Parsing failed, prepare for retry
                    last_error = result.error
                    last_raw_response = result.raw_response
                    
                    if attempt < max_retries:
                        # Exponential backoff before retry
                        backoff_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s...
                        logger.warning(
                            f"Extraction attempt {attempt + 1} failed: {last_error}. "
                            f"Retrying in {backoff_time}s..."
                        )
                        time.sleep(backoff_time)
                    else:
                        # All retries exhausted
                        result.processing_time_ms = int((time.time() - start_time) * 1000)
                        return result
                        
            except Exception as e:
                processing_time = int((time.time() - start_time) * 1000)
                
                # Handle specific API errors
                error_message = str(e)
                is_retryable = False
                
                try:
                    from google.genai import errors
                    if isinstance(e, errors.APIError):
                        error_message = f"API Error ({e.code}): {e.message}"
                        # Retry on 5xx errors or rate limits
                        is_retryable = e.code >= 500 or e.code == 429
                except ImportError:
                    pass
                
                logger.error(f"Gemini extraction attempt {attempt + 1} failed: {error_message}")
                
                if is_retryable and attempt < max_retries:
                    # Exponential backoff for API errors
                    backoff_time = (2 ** attempt) * 1.0  # 1s, 2s, 4s...
                    logger.info(f"Retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                    last_error = error_message
                else:
                    # Non-retryable error or retries exhausted
                    return GeminiExtractionResult(
                        processing_time_ms=processing_time,
                        success=False,
                        error=error_message
                    )
        
        # Should not reach here, but safety fallback
        return GeminiExtractionResult(
            processing_time_ms=int((time.time() - start_time) * 1000),
            success=False,
            error=last_error or "Unknown error after all retries"
        )
    
    def _parse_response(self, response) -> GeminiExtractionResult:
        """
        Parse Gemini response into GeminiExtractionResult.
        
        Handles both parsed (Pydantic) and raw JSON responses.
        """
        try:
            # Check if response has parsed attribute (structured output)
            if hasattr(response, 'parsed') and response.parsed:
                parsed = response.parsed
                if isinstance(parsed, dict):
                    fields = parsed.get('fields', [])
                    form_type = parsed.get('form_type', 'Unknown')
                    language = parsed.get('language', 'en')
                else:
                    # Pydantic model instance
                    fields = [f.model_dump() for f in parsed.fields]
                    form_type = parsed.form_type
                    language = parsed.language
                
                return GeminiExtractionResult(
                    fields=fields,
                    form_type=form_type,
                    language=language,
                    raw_response=response.text if hasattr(response, 'text') else str(parsed),
                    success=True
                )
            
            # Fallback: try to parse raw text as JSON
            raw_text = response.text if hasattr(response, 'text') else str(response)
            
            if not raw_text or raw_text.strip() == '':
                return GeminiExtractionResult(
                    raw_response=raw_text,
                    success=False,
                    error="Empty response from Gemini"
                )
            
            try:
                parsed = json.loads(raw_text)
                
                # Validate basic structure
                if not isinstance(parsed, dict):
                    return GeminiExtractionResult(
                        raw_response=raw_text,
                        success=False,
                        error="Response is not a JSON object"
                    )
                
                if 'fields' not in parsed:
                    return GeminiExtractionResult(
                        raw_response=raw_text,
                        success=False,
                        error="Response missing 'fields' array"
                    )
                
                return GeminiExtractionResult(
                    fields=parsed.get('fields', []),
                    form_type=parsed.get('form_type', 'Unknown'),
                    language=parsed.get('language', 'en'),
                    raw_response=raw_text,
                    success=True
                )
                
            except json.JSONDecodeError as e:
                return GeminiExtractionResult(
                    raw_response=raw_text,
                    success=False,
                    error=f"Invalid JSON: {str(e)}"
                )
                
        except Exception as e:
            return GeminiExtractionResult(
                success=False,
                error=f"Response parsing error: {str(e)}"
            )
    
    def _build_retry_prompt(
        self,
        ocr_text: str,
        custom_prompt: Optional[str],
        form_template: Optional[Dict[str, Any]],
        last_error: Optional[str],
        last_response: Optional[str],
        attempt: int
    ) -> str:
        """Build an enhanced prompt for retry attempts with error context."""
        
        parts = []
        
        # Add error correction instructions
        parts.append("=" * 60)
        parts.append("⚠️ IMPORTANT: YOUR PREVIOUS RESPONSE HAD AN ERROR")
        parts.append("=" * 60)
        parts.append(f"Error: {last_error}")
        parts.append("")
        parts.append("Please correct your response and ensure you return VALID JSON with this EXACT structure:")
        parts.append("""
{
    "form_type": "string (e.g., 'Invoice', 'Application Form')",
    "language": "string (e.g., 'en', 'hi')",
    "fields": [
        {
            "field_key": "string (the field label)",
            "field_value": "string (the extracted value)",
            "field_type": "text|number|date|email|phone|address|name|currency|unknown",
            "confidence": 0.0 to 1.0
        }
    ]
}
""")
        parts.append("=" * 60)
        parts.append("")
        
        # Add the original prompt content
        original_prompt = self._build_prompt(ocr_text, custom_prompt, form_template)
        parts.append(original_prompt)
        
        return "\n".join(parts)
    
    async def extract_from_text_async(
        self,
        ocr_text: str,
        custom_prompt: Optional[str] = None,
        form_template: Optional[Dict[str, Any]] = None,
        max_retries: int = 2
    ) -> GeminiExtractionResult:
        """
        Extract key-value pairs from OCR text asynchronously.
        
        Uses the sync client in a thread pool for true async behavior.
        Includes automatic retry logic for JSON parsing failures.
        """
        return await asyncio.to_thread(
            self.extract_from_text,
            ocr_text,
            custom_prompt,
            form_template,
            max_retries
        )
    
    # =========================================================================
    # PROMPT BUILDING
    # =========================================================================
    
    def _build_prompt(
        self,
        ocr_text: str,
        custom_prompt: Optional[str] = None,
        form_template: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the extraction prompt."""
        
        parts = []
        
        # Add custom prompt if provided
        if custom_prompt:
            parts.append(f"Additional Instructions:\n{custom_prompt}\n")
        
        # Add template info if provided
        if form_template:
            expected_fields = form_template.get('expected_fields', [])
            if expected_fields:
                fields_str = ", ".join([
                    f.get('key', f.get('name', '')) for f in expected_fields
                ])
                parts.append(f"Expected fields to look for: {fields_str}\n")
            
            template_prompt = form_template.get('extraction_prompt')
            if template_prompt:
                parts.append(f"Template Instructions:\n{template_prompt}\n")
        
        # Add the main OCR text
        parts.append("=" * 60)
        parts.append("DOCUMENT TEXT (OCR OUTPUT):")
        parts.append("=" * 60)
        parts.append(ocr_text)
        parts.append("=" * 60)
        
        # Add extraction instruction
        parts.append("\nExtract all key-value pairs from the document above.")
        parts.append("Return the results as a JSON object with 'form_type', 'language', and 'fields' array.")
        
        return "\n".join(parts)
    
    # =========================================================================
    # TOKEN COUNTING
    # =========================================================================
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.
        
        Uses server-side counting for accuracy.
        """
        try:
            self._ensure_client()
            
            result = self._client.models.count_tokens(
                model=self._model_name,
                contents=text
            )
            
            return result.total_tokens if hasattr(result, 'total_tokens') else 0
            
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
    
    async def count_tokens_async(self, text: str) -> int:
        """Count tokens asynchronously."""
        return await asyncio.to_thread(self.count_tokens, text)
    
    # =========================================================================
    # CHAT SESSIONS (For Multi-Turn Extraction)
    # =========================================================================
    
    def create_extraction_chat(self) -> Any:
        """
        Create a chat session for multi-turn extraction.
        
        Useful for:
        - Clarifying ambiguous extractions
        - Asking follow-up questions
        - Iterative refinement
        
        Returns:
            Chat session object
        """
        self._ensure_client()
        
        return self._client.chats.create(
            model=self._model_name,
            config={
                "system_instruction": self._system_instruction,
                "temperature": self._temperature
            }
        )
    
    # =========================================================================
    # STATUS & INFO
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            "client_initialized": self._client is not None,
            "model_name": self._model_name,
            "api_key_configured": bool(self._api_key),
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
            "engine": "Google Gemini"
        }
    
    def verify_api_key(self) -> bool:
        """Verify that the API key is valid."""
        try:
            self._ensure_client()
            # Try a simple operation to verify
            self._client.models.count_tokens(
                model=self._model_name,
                contents="test"
            )
            return True
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            return False


# =============================================================================
# SINGLETON & EXPORTS
# =============================================================================

# Singleton instance - use this throughout the app
gemini_service = GeminiService()


def get_gemini_status() -> Dict[str, Any]:
    """Get Gemini service status."""
    return gemini_service.get_status()


async def verify_gemini_api() -> bool:
    """Verify Gemini API key is valid."""
    return await asyncio.to_thread(gemini_service.verify_api_key)

