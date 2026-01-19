"""
RapidOCR Post-Processor: Reading Order & Line Merging
======================================================
Transforms raw OCR output into logically ordered, merged text lines.

ALGORITHM:
1. Extract bounding boxes from OCR result
2. Sort all boxes by Y-coordinate (top to bottom)
3. Group boxes into "lines" based on Y-overlap tolerance
4. Sort boxes within each line by X-coordinate (left to right)
5. Merge text within each line with spacing
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class TextBlock:
    """Represents a single detected text block with its bounding box."""
    text: str
    confidence: float
    box: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    @property
    def y_center(self) -> float:
        """Center Y coordinate of the bounding box."""
        return (self.box[0][1] + self.box[2][1]) / 2
    
    @property
    def x_left(self) -> float:
        """Left-most X coordinate."""
        return min(p[0] for p in self.box)
    
    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return abs(self.box[2][1] - self.box[0][1])


@dataclass
class MergedLine:
    """A merged line containing multiple text blocks."""
    text: str
    confidence: float  # Average confidence
    y_position: float  # Average Y for sorting
    blocks: List[TextBlock]


def parse_rapidocr_output(result) -> List[TextBlock]:
    """
    Parse RapidOCR output into TextBlock objects.
    Handles both old list format and new dataclass format.
    """
    blocks = []
    
    if result is None:
        return blocks
    
    # Get the actual items
    if hasattr(result, 'ocr_result'):
        items = result.ocr_result
    else:
        items = result
    
    if items is None:
        return blocks
    
    for item in items:
        try:
            # Handle dataclass format (newer versions)
            if hasattr(item, 'box') and hasattr(item, 'text') and hasattr(item, 'score'):
                blocks.append(TextBlock(
                    text=item.text,
                    confidence=item.score,
                    box=item.box if isinstance(item.box, list) else item.box.tolist()
                ))
            # Handle list/tuple format: [box, text, confidence]
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                box = item[0]
                text = item[1]
                conf = item[2] if len(item) > 2 else 1.0
                
                # Convert numpy arrays if needed
                if hasattr(box, 'tolist'):
                    box = box.tolist()
                
                blocks.append(TextBlock(
                    text=str(text),
                    confidence=float(conf),
                    box=box
                ))
        except Exception as e:
            print(f"  ⚠️ Failed to parse item: {e}")
            continue
    
    return blocks


def group_into_lines(blocks: List[TextBlock], y_tolerance_ratio: float = 0.5) -> List[List[TextBlock]]:
    """
    Group text blocks into lines based on Y-coordinate overlap.
    
    Args:
        blocks: List of TextBlock objects
        y_tolerance_ratio: Fraction of average height to use as tolerance
                          (0.5 = half the average character height)
    
    Returns:
        List of lines, where each line is a list of TextBlocks
    """
    if not blocks:
        return []
    
    # Sort by Y center first
    sorted_blocks = sorted(blocks, key=lambda b: b.y_center)
    
    # Calculate tolerance based on average height
    avg_height = sum(b.height for b in sorted_blocks) / len(sorted_blocks)
    y_tolerance = avg_height * y_tolerance_ratio
    
    lines: List[List[TextBlock]] = []
    current_line: List[TextBlock] = [sorted_blocks[0]]
    current_y = sorted_blocks[0].y_center
    
    for block in sorted_blocks[1:]:
        # Check if this block is on the same line
        if abs(block.y_center - current_y) <= y_tolerance:
            current_line.append(block)
            # Update current_y to be average of line
            current_y = sum(b.y_center for b in current_line) / len(current_line)
        else:
            # Start a new line
            lines.append(current_line)
            current_line = [block]
            current_y = block.y_center
    
    # Don't forget the last line
    if current_line:
        lines.append(current_line)
    
    return lines


def sort_and_merge_lines(lines: List[List[TextBlock]], space_threshold_ratio: float = 2.0) -> List[MergedLine]:
    """
    Sort blocks within each line by X coordinate and merge text.
    
    Args:
        lines: Grouped lines of TextBlocks
        space_threshold_ratio: If gap between blocks > (avg_char_width * ratio), add extra space
    
    Returns:
        List of MergedLine objects with merged text
    """
    merged_lines: List[MergedLine] = []
    
    for line_blocks in lines:
        # Sort by X (left to right)
        sorted_line = sorted(line_blocks, key=lambda b: b.x_left)
        
        # Merge text with smart spacing
        texts = []
        for i, block in enumerate(sorted_line):
            texts.append(block.text)
        
        merged_text = " ".join(texts)
        avg_confidence = sum(b.confidence for b in sorted_line) / len(sorted_line)
        avg_y = sum(b.y_center for b in sorted_line) / len(sorted_line)
        
        merged_lines.append(MergedLine(
            text=merged_text,
            confidence=avg_confidence,
            y_position=avg_y,
            blocks=sorted_line
        ))
    
    # Sort merged lines by Y position (top to bottom)
    merged_lines.sort(key=lambda l: l.y_position)
    
    return merged_lines


def process_ocr_result(
    result,
    y_tolerance_ratio: float = 0.5,
    merge_lines: bool = True
) -> List[MergedLine]:
    """
    Main function: Convert raw RapidOCR output to ordered, merged lines.
    
    Args:
        result: Raw RapidOCR output
        y_tolerance_ratio: Controls line grouping sensitivity (0.3-0.7 recommended)
        merge_lines: If True, merge text blocks on same line
    
    Returns:
        List of MergedLine objects in reading order
    """
    # Step 1: Parse raw output
    blocks = parse_rapidocr_output(result)
    
    if not blocks:
        return []
    
    # Step 2: Group into lines
    lines = group_into_lines(blocks, y_tolerance_ratio)
    
    # Step 3: Sort within lines and merge
    merged = sort_and_merge_lines(lines)
    
    return merged


def format_merged_output(merged_lines: List[MergedLine], show_confidence: bool = False) -> str:
    """
    Format merged lines into a clean string output.
    """
    output = []
    for i, line in enumerate(merged_lines, 1):
        if show_confidence:
            output.append(f"{i:02d}. [{line.confidence:.2f}] {line.text}")
        else:
            output.append(f"{i:02d}. {line.text}")
    return "\n".join(output)


# ============================================================
# CONVENIENCE FUNCTION FOR DIRECT USE
# ============================================================

def extract_text_ordered(result, y_tolerance: float = 0.5) -> str:
    """
    One-liner to get clean, ordered text from RapidOCR result.
    
    Usage:
        result = engine(img_path)
        text = extract_text_ordered(result)
        print(text)
    """
    merged = process_ocr_result(result, y_tolerance_ratio=y_tolerance)
    return format_merged_output(merged, show_confidence=False)
