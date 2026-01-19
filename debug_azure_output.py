"""
Debug Script for Azure Document Intelligence Output
===================================================
This script processes a single test image and shows the exact Azure response structure.

Usage:
    python debug_azure_output.py path/to/test/image.png
"""

import sys
import json
from pathlib import Path
import asyncio

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from services.ocr_service import OCRService
from config import settings


async def debug_azure_output(image_path: str):
    """Process image and show Azure output structure."""
    
    print("="*60)
    print("AZURE DOCUMENT INTELLIGENCE DEBUG")
    print("="*60)
    print(f"\nProcessing: {image_path}\n")
    
    # Initialize OCR service
    ocr_service = OCRService()
    
    # Process the image
    result = await ocr_service.process_document(image_path, "png")
    
    if not result.success:
        print(f"❌ OCR Failed: {result.error}")
        return
    
    print(f"✅ OCR Succeeded!")
    print(f"   Pages: {result.total_pages}")
    print(f"   Processing time: {result.total_processing_time_ms}ms")
    print(f"\n" + "="*60)
    print("LAYOUT BOXES")
    print("="*60)
    print(f"Total combined_layout_boxes: {len(result.combined_layout_boxes)}")
    
    if result.combined_layout_boxes:
        print(f"\n📦 Sample boxes (first 5):")
        for i, box in enumerate(result.combined_layout_boxes[:5]):
            print(f"\n  Box {i+1}:")
            print(f"    Type: {box.get('type')}")
            print(f"    Content: {box.get('content', 'N/A')[:50]}")
            print(f"    Page: {box.get('page_number')}")
            print(f"    Polygon length: {len(box.get('polygon', []))}")
            print(f"    Polygon sample: {box.get('polygon', [])[:8]}...")
    else:
        print("⚠️  NO LAYOUT BOXES EXTRACTED!")
    
    print(f"\n" + "="*60)
    print("PROCESSED IMAGES")
    print("="*60)
    print(f"Total pages with processed images: {len(result.pages)}")
    
    for i, page in enumerate(result.pages, 1):
        print(f"\n  Page {i}:")
        print(f"    Has processed_image_bytes: {page.processed_image_bytes is not None}")
        if page.processed_image_bytes:
            print(f"    Image size: {len(page.processed_image_bytes)} bytes")
        print(f"    Page dimensions: {page.page_width_inches}\" x {page.page_height_inches}\"")
        print(f"    Layout boxes for this page: {len(page.layout_boxes)}")
    
    print(f"\n" + "="*60)
    print("RAW PAGE DATA STRUCTURE")
    print("="*60)
    
    # Show structure of first page
    if result.pages:
        first_page = result.pages[0]
        print(f"\nFirst page object structure:")
        print(f"  - markdown: {len(first_page.markdown)} chars")
        print(f"  - html: {len(first_page.html)} chars")  
        print(f"  - layout_boxes: {len(first_page.layout_boxes)} items")
        print(f"  - processed_image_bytes: {len(first_page.processed_image_bytes) if first_page.processed_image_bytes else 0} bytes")
        print(f"  - page_width_inches: {first_page.page_width_inches}")
        print(f"  - page_height_inches: {first_page.page_height_inches}")
    
    print(f"\n" + "="*60)
    print("SAVE AS JSON FOR INSPECTION")
    print("="*60)
    
    # Save detailed output to JSON
    output_data = {
        "success": result.success,
        "total_pages": result.total_pages,
        "combined_layout_boxes_count": len(result.combined_layout_boxes),
        "combined_layout_boxes_sample": result.combined_layout_boxes[:10] if result.combined_layout_boxes else [],
        "pages": [
            {
                "page_number": page.page_number,
                "layout_boxes_count": len(page.layout_boxes),
                "has_processed_image": page.processed_image_bytes is not None,
                "page_width_inches": page.page_width_inches,
                "page_height_inches": page.page_height_inches,
            }
            for page in result.pages
        ]
    }
    
    output_file = Path("azure_debug_output.json")
    output_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved detailed output to: {output_file.absolute()}")
    print(f"\nExamine this file to see the exact structure of extracted data.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_azure_output.py path/to/image.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)
    
    asyncio.run(debug_azure_output(image_path))
