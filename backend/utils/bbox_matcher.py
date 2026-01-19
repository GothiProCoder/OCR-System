"""
Bounding Box Matcher
====================
Links Gemini-extracted field text to OCR layout boxes using fuzzy matching.

This utility finds the best matching OCR bounding boxes for extracted
KEY-VALUE pairs by using a multi-strategy approach:
1. Exact match (fastest)
2. Fuzzy match using SequenceMatcher
3. Multi-word union (for concatenated values)

Reference: Azure Document Intelligence output format
- Polygons are [x1,y1, x2,y2, x3,y3, x4,y4] in inches
- Origin is top-left (y increases downward)
"""

from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
import re
import logging

logger = logging.getLogger(__name__)


class BoundingBoxMatcher:
    """
    Matches Gemini-extracted field values to OCR layout boxes.
    
    Usage:
        matcher = BoundingBoxMatcher(layout_data)
        key_bbox, value_bbox = matcher.find_key_value_pair("Full Name", "John Smith")
    """
    
    # Match threshold: 85% similarity required for fuzzy matching
    FUZZY_THRESHOLD = 0.85
    
    def __init__(self, layout_data: List[Dict[str, Any]]):
        """
        Initialize with OCR layout data.
        
        Args:
            layout_data: List of OCR boxes with {type, content, polygon, page_number}
        """
        self.layout_data = layout_data or []
        
        # Pre-filter by type for faster lookup
        self._lines = [b for b in self.layout_data if b.get("type") == "line"]
        self._words = [b for b in self.layout_data if b.get("type") == "word"]
        
        logger.debug(f"BoundingBoxMatcher initialized with {len(self._lines)} lines, {len(self._words)} words")
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text for comparison.
        
        - Strips whitespace
        - Converts to lowercase
        - Collapses multiple spaces
        """
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text.strip().lower())
    
    @staticmethod
    def fuzzy_ratio(a: str, b: str) -> float:
        """
        Calculate similarity ratio between two strings.
        
        Uses Python's SequenceMatcher which is optimized for string matching.
        Returns value between 0.0 (no match) and 1.0 (exact match).
        """
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
    
    def find_match(
        self, 
        target_text: str, 
        page_number: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find best matching OCR box for target text.
        
        Strategy:
        1. Exact line match (confidence = 1.0)
        2. Fuzzy line match (confidence = ratio)
        3. Multi-word union (confidence = avg of matched words)
        
        Args:
            target_text: The text to find in OCR output
            page_number: Optional page filter
            
        Returns:
            Dict with {polygon, matched_text, confidence, page} or None
        """
        if not target_text or not target_text.strip():
            return None
        
        target_norm = self.normalize(target_text)
        
        # Strategy 1: Exact line match (fastest, highest confidence)
        for line in self._lines:
            if page_number and line.get("page_number") != page_number:
                continue
            line_content = self.normalize(line.get("content", ""))
            if line_content == target_norm:
                logger.debug(f"Exact match found for '{target_text[:30]}...'")
                return {
                    "polygon": line.get("polygon", []),
                    "matched_text": line.get("content", ""),
                    "confidence": 1.0,
                    "page": line.get("page_number", 1)
                }
        
        # Strategy 2: Fuzzy line match
        best_match = None
        best_score = 0.0
        
        for line in self._lines:
            if page_number and line.get("page_number") != page_number:
                continue
            line_content = self.normalize(line.get("content", ""))
            
            # Check if target is contained in line or vice versa
            if target_norm in line_content or line_content in target_norm:
                score = self.fuzzy_ratio(target_norm, line_content)
                if score > best_score:
                    best_match = line
                    best_score = max(score, 0.9)  # Boost for containment
            else:
                score = self.fuzzy_ratio(target_norm, line_content)
                if score >= self.FUZZY_THRESHOLD and score > best_score:
                    best_match = line
                    best_score = score
        
        if best_match:
            logger.debug(f"Fuzzy match found for '{target_text[:30]}...' with score {best_score:.2f}")
            return {
                "polygon": best_match.get("polygon", []),
                "matched_text": best_match.get("content", ""),
                "confidence": best_score,
                "page": best_match.get("page_number", 1)
            }
        
        # Strategy 3: Multi-word union (for concatenated values)
        word_union = self._find_word_union(target_text, page_number)
        if word_union:
            logger.debug(f"Word union found for '{target_text[:30]}...'")
            return word_union
        
        logger.debug(f"No match found for '{target_text[:30]}...'")
        return None
    
    def _find_word_union(
        self, 
        target_text: str, 
        page_number: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find multiple words and compute union bounding box.
        
        Used when Gemini concatenates multiple OCR words into one value.
        """
        words = target_text.split()
        if not words or len(words) < 2:
            # For single words, try direct word match
            if len(words) == 1:
                return self._find_single_word(words[0], page_number)
            return None
        
        matched_boxes = []
        matched_texts = []
        
        for word in words:
            word_norm = self.normalize(word)
            if len(word_norm) < 2:  # Skip very short words
                continue
                
            for w in self._words:
                if page_number and w.get("page_number") != page_number:
                    continue
                w_content = self.normalize(w.get("content", ""))
                
                # Exact or near-exact word match
                if w_content == word_norm or self.fuzzy_ratio(w_content, word_norm) >= 0.9:
                    matched_boxes.append(w)
                    matched_texts.append(w.get("content", ""))
                    break  # Take first match per word
        
        if not matched_boxes:
            return None
        
        # Need at least 50% of words matched
        if len(matched_boxes) < len(words) * 0.5:
            return None
        
        # Compute union polygon
        union_polygon = self._compute_union(matched_boxes)
        matched_text = " ".join(matched_texts)
        avg_conf = len(matched_boxes) / len(words)  # Match ratio as confidence
        
        return {
            "polygon": union_polygon,
            "matched_text": matched_text,
            "confidence": min(avg_conf, 0.95),  # Cap at 0.95 for unions
            "page": matched_boxes[0].get("page_number", 1)
        }
    
    def _find_single_word(
        self,
        word: str,
        page_number: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Find a single word in the word-level boxes."""
        word_norm = self.normalize(word)
        
        for w in self._words:
            if page_number and w.get("page_number") != page_number:
                continue
            w_content = self.normalize(w.get("content", ""))
            
            if w_content == word_norm:
                return {
                    "polygon": w.get("polygon", []),
                    "matched_text": w.get("content", ""),
                    "confidence": 1.0,
                    "page": w.get("page_number", 1)
                }
            elif self.fuzzy_ratio(w_content, word_norm) >= self.FUZZY_THRESHOLD:
                return {
                    "polygon": w.get("polygon", []),
                    "matched_text": w.get("content", ""),
                    "confidence": self.fuzzy_ratio(w_content, word_norm),
                    "page": w.get("page_number", 1)
                }
        
        return None
    
    def _compute_union(self, boxes: List[Dict[str, Any]]) -> List[float]:
        """
        Compute union bounding box from multiple word boxes.
        
        For quadrilateral polygons [x1,y1, x2,y2, x3,y3, x4,y4]:
        - Take min(all_x) for left edge
        - Take max(all_x) for right edge
        - Take min(all_y) for top edge
        - Take max(all_y) for bottom edge
        
        Returns axis-aligned bounding rectangle as quadrilateral.
        """
        all_x, all_y = [], []
        
        for box in boxes:
            polygon = box.get("polygon", [])
            for i in range(0, len(polygon), 2):
                if i + 1 < len(polygon):
                    all_x.append(polygon[i])
                    all_y.append(polygon[i + 1])
        
        if not all_x or not all_y:
            return []
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Return as quadrilateral: top-left, top-right, bottom-right, bottom-left
        return [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
    
    def find_key_value_pair(
        self,
        field_key: str,
        field_value: str,
        page_number: Optional[int] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Find bounding boxes for both KEY and VALUE.
        
        Args:
            field_key: The field label (e.g., "Full Name")
            field_value: The field value (e.g., "John Smith")
            page_number: Optional page filter
            
        Returns:
            Tuple of (key_bbox, value_bbox) - either can be None if not found
        """
        key_bbox = self.find_match(field_key, page_number)
        value_bbox = self.find_match(field_value, page_number)
        
        return key_bbox, value_bbox
