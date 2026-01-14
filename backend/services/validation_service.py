"""
Validation Service - Field Validation & Type Checking
======================================================
Production-grade service for validating extracted field values based on
their detected types. Integrates with LangGraph workflow and can run
automatically after extraction or standalone.

Features:
- Type-specific validators (email, phone, date, number, currency, etc.)
- Confidence-based validation rules
- Auto-correction suggestions for common issues
- LangGraph @task for workflow integration
- Batch validation with database updates
- Comprehensive validation results with actionable messages

Usage:
    from services.validation_service import validation_service
    
    # Validate single field
    result = validation_service.validate_field(
        field_key="Email",
        field_value="john@example.com",
        field_type="email",
        confidence=0.92
    )
    
    # Batch validate extraction
    batch_result = await validation_service.validate_extraction(extraction_id)
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID

# LangGraph task decorator
from langgraph.func import task
from langgraph.types import RetryPolicy

# Database imports
from database.crud import (
    extraction_crud,
    field_crud,
    processing_log_crud
)
from database.models import (
    FieldType,
    ProcessingStep,
    LogStatus
)
from database.connection import get_async_db

# Configuration
from config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES - Validation Results
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validating a single field."""
    field_id: Optional[UUID] = None
    field_key: str = ""
    is_valid: bool = True
    message: Optional[str] = None
    severity: str = "info"  # info, warning, error
    corrected_value: Optional[str] = None  # Suggested auto-correction
    confidence_level: str = "unknown"  # high, medium, low
    needs_review: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_id": str(self.field_id) if self.field_id else None,
            "field_key": self.field_key,
            "is_valid": self.is_valid,
            "message": self.message,
            "severity": self.severity,
            "corrected_value": self.corrected_value,
            "confidence_level": self.confidence_level,
            "needs_review": self.needs_review
        }


@dataclass
class BatchValidationResult:
    """Result of validating multiple fields."""
    total_fields: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    warning_count: int = 0
    needs_review_count: int = 0
    field_results: List[ValidationResult] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    
    @property
    def validation_rate(self) -> float:
        """Percentage of valid fields."""
        if self.total_fields == 0:
            return 0.0
        return (self.valid_count / self.total_fields) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_fields": self.total_fields,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "warning_count": self.warning_count,
            "needs_review_count": self.needs_review_count,
            "validation_rate": round(self.validation_rate, 2),
            "field_results": [r.to_dict() for r in self.field_results],
            "success": self.success,
            "error": self.error
        }


# =============================================================================
# VALIDATION PATTERNS - Regex & Rules
# =============================================================================

# Email validation
EMAIL_PATTERN = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

# Phone validation patterns (flexible for multiple formats)
PHONE_PATTERNS = [
    re.compile(r'^\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'),  # US format
    re.compile(r'^\+91[\s-]?[6-9][0-9]{9}$'),  # India format
    re.compile(r'^\+?[\d\s\-\(\)\.]{7,20}$'),  # Generic international
]

# Date formats to try parsing
DATE_FORMATS = [
    '%Y-%m-%d',      # 2024-01-15
    '%d/%m/%Y',      # 15/01/2024
    '%m/%d/%Y',      # 01/15/2024
    '%d-%m-%Y',      # 15-01-2024
    '%d %b %Y',      # 15 Jan 2024
    '%d %B %Y',      # 15 January 2024
    '%B %d, %Y',     # January 15, 2024
    '%b %d, %Y',     # Jan 15, 2024
    '%Y/%m/%d',      # 2024/01/15
]

# Currency patterns
CURRENCY_PATTERN = re.compile(
    r'^[\$\£\€\₹\¥]?\s*[\d,]+\.?\d*$|'  # $1,234.56 or ₹1234
    r'^[\d,]+\.?\d*\s*[\$\£\€\₹\¥]?$'   # 1234.56$ or 1234₹
)

# Number patterns  
NUMBER_PATTERN = re.compile(r'^-?[\d,]+\.?\d*$')

# Checkbox/boolean values
CHECKBOX_VALUES = {
    # True values
    'yes', 'true', 'checked', '1', 'on', 'x', '✓', '✔', 'y',
    # False values  
    'no', 'false', 'unchecked', '0', 'off', '', 'n'
}

# Name validation
NAME_PATTERN = re.compile(r'^[a-zA-Z\s\.\-\']+$')


# =============================================================================
# CONFIDENCE THRESHOLDS (from config)
# =============================================================================

CONFIDENCE_HIGH = getattr(settings, 'CONFIDENCE_HIGH_THRESHOLD', 0.85)
CONFIDENCE_MEDIUM = getattr(settings, 'CONFIDENCE_MEDIUM_THRESHOLD', 0.60)


def get_confidence_level(confidence: float) -> str:
    """Categorize confidence score."""
    if confidence >= CONFIDENCE_HIGH:
        return "high"
    elif confidence >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


# =============================================================================
# TYPE-SPECIFIC VALIDATORS
# =============================================================================

def validate_email(value: str, confidence: float) -> ValidationResult:
    """Validate email format."""
    if not value or not value.strip():
        return ValidationResult(
            is_valid=False,
            message="Email field is empty",
            severity="error",
            confidence_level=get_confidence_level(confidence)
        )
    
    value = value.strip().lower()
    
    if EMAIL_PATTERN.match(value):
        return ValidationResult(
            is_valid=True,
            message="Valid email format",
            severity="info",
            confidence_level=get_confidence_level(confidence)
        )
    
    # Try to suggest correction
    corrected = None
    if ' ' in value:
        corrected = value.replace(' ', '')
        if EMAIL_PATTERN.match(corrected):
            return ValidationResult(
                is_valid=False,
                message="Email contains spaces - did you mean: " + corrected,
                severity="warning",
                corrected_value=corrected,
                confidence_level=get_confidence_level(confidence)
            )
    
    return ValidationResult(
        is_valid=False,
        message="Invalid email format",
        severity="error",
        confidence_level=get_confidence_level(confidence)
    )


def validate_phone(value: str, confidence: float) -> ValidationResult:
    """Validate phone number format."""
    if not value or not value.strip():
        return ValidationResult(
            is_valid=False,
            message="Phone number is empty",
            severity="error",
            confidence_level=get_confidence_level(confidence)
        )
    
    cleaned = value.strip()
    
    # Check against known patterns
    for pattern in PHONE_PATTERNS:
        if pattern.match(cleaned):
            return ValidationResult(
                is_valid=True,
                message="Valid phone format",
                severity="info",
                confidence_level=get_confidence_level(confidence)
            )
    
    # Check if it has enough digits
    digits_only = re.sub(r'\D', '', cleaned)
    if len(digits_only) >= 7 and len(digits_only) <= 15:
        return ValidationResult(
            is_valid=True,
            message="Phone number has valid digit count",
            severity="info",
            confidence_level=get_confidence_level(confidence),
            needs_review=confidence < CONFIDENCE_HIGH
        )
    
    return ValidationResult(
        is_valid=False,
        message=f"Invalid phone number (found {len(digits_only)} digits, expected 7-15)",
        severity="error",
        confidence_level=get_confidence_level(confidence)
    )


def validate_date(value: str, confidence: float) -> ValidationResult:
    """Validate date format and parse attempt."""
    if not value or not value.strip():
        return ValidationResult(
            is_valid=False,
            message="Date field is empty",
            severity="error",
            confidence_level=get_confidence_level(confidence)
        )
    
    cleaned = value.strip()
    
    # Try parsing with known formats
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            # Format to ISO standard
            corrected = parsed.strftime('%Y-%m-%d')
            return ValidationResult(
                is_valid=True,
                message=f"Valid date: {corrected}",
                severity="info",
                corrected_value=corrected if corrected != cleaned else None,
                confidence_level=get_confidence_level(confidence)
            )
        except ValueError:
            continue
    
    # Check if it looks like a date (has numbers and separators)
    if re.search(r'\d+[\/\-\.]\d+[\/\-\.]\d+', cleaned):
        return ValidationResult(
            is_valid=False,
            message="Date format not recognized - please verify",
            severity="warning",
            needs_review=True,
            confidence_level=get_confidence_level(confidence)
        )
    
    return ValidationResult(
        is_valid=False,
        message="Invalid date format",
        severity="error",
        confidence_level=get_confidence_level(confidence)
    )


def validate_number(value: str, confidence: float) -> ValidationResult:
    """Validate numeric value."""
    if not value or not value.strip():
        return ValidationResult(
            is_valid=False,
            message="Number field is empty",
            severity="error",
            confidence_level=get_confidence_level(confidence)
        )
    
    cleaned = value.strip().replace(',', '').replace(' ', '')
    
    if NUMBER_PATTERN.match(cleaned):
        try:
            float(cleaned)
            return ValidationResult(
                is_valid=True,
                message="Valid number",
                severity="info",
                confidence_level=get_confidence_level(confidence)
            )
        except ValueError:
            pass
    
    # Check if mostly numeric
    digits = sum(c.isdigit() for c in cleaned)
    if digits / max(len(cleaned), 1) > 0.8:
        return ValidationResult(
            is_valid=False,
            message="Value appears to be a number but has invalid characters",
            severity="warning",
            needs_review=True,
            confidence_level=get_confidence_level(confidence)
        )
    
    return ValidationResult(
        is_valid=False,
        message="Invalid number format",
        severity="error",
        confidence_level=get_confidence_level(confidence)
    )


def validate_currency(value: str, confidence: float) -> ValidationResult:
    """Validate currency value."""
    if not value or not value.strip():
        return ValidationResult(
            is_valid=False,
            message="Currency field is empty",
            severity="error",
            confidence_level=get_confidence_level(confidence)
        )
    
    cleaned = value.strip()
    
    if CURRENCY_PATTERN.match(cleaned):
        return ValidationResult(
            is_valid=True,
            message="Valid currency format",
            severity="info",
            confidence_level=get_confidence_level(confidence)
        )
    
    # Try to extract numeric value
    numeric = re.sub(r'[^\d.,]', '', cleaned)
    if numeric and NUMBER_PATTERN.match(numeric.replace(',', '')):
        return ValidationResult(
            is_valid=True,
            message="Currency value extracted",
            severity="info",
            corrected_value=numeric,
            confidence_level=get_confidence_level(confidence)
        )
    
    return ValidationResult(
        is_valid=False,
        message="Invalid currency format",
        severity="error",
        confidence_level=get_confidence_level(confidence)
    )


def validate_checkbox(value: str, confidence: float) -> ValidationResult:
    """Validate checkbox/boolean value."""
    if value is None:
        value = ""
    
    cleaned = value.strip().lower()
    
    if cleaned in CHECKBOX_VALUES:
        return ValidationResult(
            is_valid=True,
            message="Valid checkbox value",
            severity="info",
            confidence_level=get_confidence_level(confidence)
        )
    
    return ValidationResult(
        is_valid=False,
        message=f"Unrecognized checkbox value: '{value}' (expected yes/no, true/false, etc.)",
        severity="warning",
        needs_review=True,
        confidence_level=get_confidence_level(confidence)
    )


def validate_name(value: str, confidence: float) -> ValidationResult:
    """Validate name field."""
    if not value or not value.strip():
        return ValidationResult(
            is_valid=False,
            message="Name field is empty",
            severity="error",
            confidence_level=get_confidence_level(confidence)
        )
    
    cleaned = value.strip()
    
    # Check minimum length
    if len(cleaned) < 2:
        return ValidationResult(
            is_valid=False,
            message="Name too short",
            severity="error",
            confidence_level=get_confidence_level(confidence)
        )
    
    # Check for numbers (names shouldn't have numbers)
    if re.search(r'\d', cleaned):
        return ValidationResult(
            is_valid=False,
            message="Name contains numbers",
            severity="warning",
            needs_review=True,
            confidence_level=get_confidence_level(confidence)
        )
    
    # Check pattern (letters, spaces, hyphens, apostrophes, periods)
    if NAME_PATTERN.match(cleaned):
        return ValidationResult(
            is_valid=True,
            message="Valid name format",
            severity="info",
            confidence_level=get_confidence_level(confidence)
        )
    
    # Still might be valid for international names
    return ValidationResult(
        is_valid=True,
        message="Name contains special characters - please verify",
        severity="info",
        needs_review=confidence < CONFIDENCE_HIGH,
        confidence_level=get_confidence_level(confidence)
    )


def validate_address(value: str, confidence: float) -> ValidationResult:
    """Validate address field."""
    if not value or not value.strip():
        return ValidationResult(
            is_valid=False,
            message="Address field is empty",
            severity="error",
            confidence_level=get_confidence_level(confidence)
        )
    
    cleaned = value.strip()
    
    # Check minimum reasonable length for address
    if len(cleaned) < 10:
        return ValidationResult(
            is_valid=False,
            message="Address seems too short",
            severity="warning",
            needs_review=True,
            confidence_level=get_confidence_level(confidence)
        )
    
    # Valid if has alphanumeric content
    return ValidationResult(
        is_valid=True,
        message="Address format accepted",
        severity="info",
        needs_review=confidence < CONFIDENCE_MEDIUM,
        confidence_level=get_confidence_level(confidence)
    )


def validate_text(value: str, confidence: float) -> ValidationResult:
    """Validate generic text field."""
    if not value or not value.strip():
        # Empty text might be valid in some cases
        return ValidationResult(
            is_valid=True,
            message="Text field is empty",
            severity="info",
            needs_review=confidence < CONFIDENCE_HIGH,
            confidence_level=get_confidence_level(confidence)
        )
    
    # Generic text is always valid
    return ValidationResult(
        is_valid=True,
        message="Text field accepted",
        severity="info",
        needs_review=confidence < CONFIDENCE_MEDIUM,
        confidence_level=get_confidence_level(confidence)
    )


def validate_signature(value: str, confidence: float) -> ValidationResult:
    """Validate signature field (presence check mainly)."""
    if not value or not value.strip():
        return ValidationResult(
            is_valid=False,
            message="Signature not detected",
            severity="warning",
            needs_review=True,
            confidence_level=get_confidence_level(confidence)
        )
    
    return ValidationResult(
        is_valid=True,
        message="Signature detected",
        severity="info",
        needs_review=confidence < CONFIDENCE_MEDIUM,
        confidence_level=get_confidence_level(confidence)
    )


def validate_table(value: str, confidence: float) -> ValidationResult:
    """Validate table field (presence check)."""
    if not value or not value.strip():
        return ValidationResult(
            is_valid=False,
            message="Table data is empty",
            severity="warning",
            needs_review=True,
            confidence_level=get_confidence_level(confidence)
        )
    
    return ValidationResult(
        is_valid=True,
        message="Table data present",
        severity="info",
        needs_review=True,  # Tables always need review
        confidence_level=get_confidence_level(confidence)
    )


def validate_unknown(value: str, confidence: float) -> ValidationResult:
    """Validate unknown field type - flag for review."""
    return ValidationResult(
        is_valid=True,
        message="Field type unknown - please review",
        severity="info",
        needs_review=True,
        confidence_level=get_confidence_level(confidence)
    )


# =============================================================================
# VALIDATOR REGISTRY
# =============================================================================

VALIDATORS: Dict[str, Callable[[str, float], ValidationResult]] = {
    "email": validate_email,
    "phone": validate_phone,
    "date": validate_date,
    "number": validate_number,
    "currency": validate_currency,
    "checkbox": validate_checkbox,
    "name": validate_name,
    "address": validate_address,
    "text": validate_text,
    "signature": validate_signature,
    "table": validate_table,
    "unknown": validate_unknown,
}


# =============================================================================
# LANGGRAPH TASK - Workflow Integration
# =============================================================================

validation_retry_policy = RetryPolicy(
    max_attempts=2,
    initial_interval=0.5,
    backoff_factor=2.0,
    retry_on=Exception
)


@task(retry=validation_retry_policy)
def run_validation_task(fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    LangGraph task for batch field validation.
    
    Can be integrated into extraction_workflow or called standalone.
    
    Args:
        fields: List of field dicts with keys:
            - field_id: UUID (optional)
            - field_key: str
            - field_value: str
            - field_type: str
            - confidence: float
            
    Returns:
        BatchValidationResult as dict
    """
    try:
        service = ValidationService()
        result = service.validate_fields(fields)
        return result.to_dict()
    except Exception as e:
        logger.exception(f"Validation task error: {e}")
        return BatchValidationResult(
            success=False,
            error=str(e)
        ).to_dict()


# =============================================================================
# VALIDATION SERVICE CLASS
# =============================================================================

class ValidationService:
    """
    Production-grade field validation service.
    
    Provides type-specific validation with confidence-based rules,
    auto-correction suggestions, and database integration.
    
    Features:
    - Singleton pattern for efficiency
    - Type-specific validators
    - Confidence-based validation rules
    - Batch validation with DB updates
    - LangGraph workflow integration
    """
    
    _instance: Optional["ValidationService"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "ValidationService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ValidationService._initialized:
            return
        
        self._validators = VALIDATORS
        ValidationService._initialized = True
        logger.info("ValidationService initialized")
    
    def get_validator(self, field_type: str) -> Callable[[str, float], ValidationResult]:
        """Get appropriate validator for field type."""
        field_type_lower = field_type.lower() if field_type else "unknown"
        return self._validators.get(field_type_lower, validate_unknown)
    
    def validate_field(
        self,
        field_key: str,
        field_value: str,
        field_type: str,
        confidence: float = 0.85,
        field_id: Optional[UUID] = None
    ) -> ValidationResult:
        """
        Validate a single extracted field.
        
        Args:
            field_key: Field label/name
            field_value: Extracted value
            field_type: Type of field (email, phone, date, etc.)
            confidence: OCR/LLM confidence score (0-1)
            field_id: Optional UUID for tracking
            
        Returns:
            ValidationResult with validation status and message
        """
        validator = self.get_validator(field_type)
        result = validator(field_value or "", confidence)
        
        # Add field metadata
        result.field_id = field_id
        result.field_key = field_key
        
        # Low confidence always needs review
        if confidence < CONFIDENCE_MEDIUM:
            result.needs_review = True
            if result.message:
                result.message += " (low confidence - needs review)"
            else:
                result.message = "Low confidence - needs review"
        
        return result
    
    def validate_fields(
        self,
        fields: List[Dict[str, Any]]
    ) -> BatchValidationResult:
        """
        Validate multiple fields.
        
        Args:
            fields: List of field dicts with field_key, field_value, 
                   field_type, confidence, and optionally field_id
                   
        Returns:
            BatchValidationResult with all results
        """
        results = []
        valid_count = 0
        invalid_count = 0
        warning_count = 0
        needs_review_count = 0
        
        for field_data in fields:
            result = self.validate_field(
                field_key=field_data.get("field_key", ""),
                field_value=field_data.get("field_value", ""),
                field_type=field_data.get("field_type", "text"),
                confidence=field_data.get("confidence", 0.85),
                field_id=field_data.get("field_id") or field_data.get("id")
            )
            
            results.append(result)
            
            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1
            
            if result.severity == "warning":
                warning_count += 1
            
            if result.needs_review:
                needs_review_count += 1
        
        return BatchValidationResult(
            total_fields=len(fields),
            valid_count=valid_count,
            invalid_count=invalid_count,
            warning_count=warning_count,
            needs_review_count=needs_review_count,
            field_results=results,
            success=True
        )
    
    async def validate_extraction(
        self,
        extraction_id: UUID,
        update_database: bool = True
    ) -> BatchValidationResult:
        """
        Validate all fields in an extraction and optionally update database.
        
        Args:
            extraction_id: UUID of the extraction
            update_database: If True, update is_valid and validation_message in DB
            
        Returns:
            BatchValidationResult with all field validations
        """
        try:
            async with get_async_db() as db:
                # Get extraction with fields
                extraction = await extraction_crud.get_with_fields(db, extraction_id)
                
                if not extraction:
                    return BatchValidationResult(
                        success=False,
                        error=f"Extraction {extraction_id} not found"
                    )
                
                # Prepare field data for validation
                fields_data = []
                for field in extraction.fields:
                    fields_data.append({
                        "field_id": field.id,
                        "field_key": field.field_key,
                        "field_value": field.field_value,
                        "field_type": field.field_type.value if field.field_type else "text",
                        "confidence": field.confidence or 0.85
                    })
                
                # Validate all fields
                batch_result = self.validate_fields(fields_data)
                
                # Update database if requested
                if update_database and batch_result.success:
                    updates = []
                    for result in batch_result.field_results:
                        if result.field_id:
                            updates.append({
                                "id": result.field_id,
                                "is_valid": result.is_valid,
                                "validation_message": result.message
                            })
                    
                    if updates:
                        await field_crud.bulk_update(db, updates)
                    
                    # Log validation step
                    await processing_log_crud.log_step(
                        db,
                        document_id=extraction.document_id,
                        step=ProcessingStep.VALIDATION,
                        status=LogStatus.COMPLETED,
                        message=f"Validated {batch_result.total_fields} fields: "
                               f"{batch_result.valid_count} valid, "
                               f"{batch_result.invalid_count} invalid, "
                               f"{batch_result.needs_review_count} need review",
                        extraction_id=extraction_id,
                        details=batch_result.to_dict()
                    )
                    
                    await db.commit()
                
                logger.info(
                    f"Validation completed for extraction {extraction_id}: "
                    f"{batch_result.valid_count}/{batch_result.total_fields} valid"
                )
                
                return batch_result
                
        except Exception as e:
            logger.exception(f"Validation error for extraction {extraction_id}: {e}")
            return BatchValidationResult(
                success=False,
                error=str(e)
            )
    
    async def validate_before_finalization(
        self,
        extraction_id: UUID
    ) -> Tuple[bool, BatchValidationResult]:
        """
        Validate extraction before finalization (saving to DB permanently).
        
        Returns:
            Tuple of (can_finalize: bool, validation_result: BatchValidationResult)
        """
        result = await self.validate_extraction(extraction_id, update_database=True)
        
        # Can finalize if all fields are valid or have only warnings
        can_finalize = result.success and result.invalid_count == 0
        
        if not can_finalize and result.success:
            logger.warning(
                f"Extraction {extraction_id} has {result.invalid_count} invalid fields"
            )
        
        return can_finalize, result
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status for health checks."""
        return {
            "service": "ValidationService",
            "status": "ready",
            "validators_available": list(self._validators.keys()),
            "confidence_high_threshold": CONFIDENCE_HIGH,
            "confidence_medium_threshold": CONFIDENCE_MEDIUM
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

validation_service = ValidationService()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ValidationService",
    "validation_service",
    "ValidationResult",
    "BatchValidationResult",
    "run_validation_task",
    "validate_email",
    "validate_phone",
    "validate_date",
    "validate_number",
    "validate_currency",
    "validate_checkbox",
    "validate_name",
    "validate_address",
    "validate_text"
]
