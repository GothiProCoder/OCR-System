"""
Interactive Document Viewer
============================
Displays processed document images with SVG bounding box overlays.
Supports field highlighting, auto-scroll to page, and zoom.

Features:
- Display processed (deskewed) images from backend
- Draw SVG polygon overlays for bounding boxes
- Highlight active field when clicked in KVEditor
- Auto-scroll to correct page for multi-page documents
- Slight zoom/focus around active bounding box
"""

import streamlit as st
import streamlit.components.v1 as components
import base64
from typing import Dict, Any, List, Optional
from services.api import APIClient


class InteractiveDocumentViewer:
    """
    Component for viewing processed documents with bounding box overlays.
    
    Usage:
        viewer = InteractiveDocumentViewer(
            doc_id="uuid",
            layout_data=[...],
            page_dimensions={...},
            processed_image_paths={...}
        )
        viewer.render(selected_field_id="field-uuid")
    """
    
    def __init__(
        self,
        doc_id: str,
        layout_data: List[Dict[str, Any]],
        page_dimensions: Dict[str, Dict[str, float]],
        processed_image_paths: Dict[str, str],
        total_pages: int = 1
    ):
        self.doc_id = doc_id
        self.layout_data = layout_data or []
        self.page_dimensions = page_dimensions or {}
        self.processed_image_paths = processed_image_paths or {}
        self.total_pages = total_pages
    
    def _get_image_as_base64(self, page_num: int) -> Optional[str]:
        """Fetch processed image and convert to base64 for embedding."""
        image_bytes = APIClient.get_processed_image_bytes(self.doc_id, page_num)
        if image_bytes:
            return base64.b64encode(image_bytes).decode('utf-8')
        return None
    
    def _azure_polygon_to_svg_points(
        self,
        polygon: List[float],
        page_width_inches: float,
        page_height_inches: float
    ) -> str:
        """
        Format Azure polygon coordinates as SVG points string.
        
        Azure origin is Top-Left (same as SVG).
        Returns raw coordinates to be used with viewBox="0 0 W H".
        
        Args:
            polygon: List of [x1, y1, x2, y2, ...] coordinates
            page_width_inches: Page width (unused but kept for API compat)
            page_height_inches: Page height (unused but kept for API compat)
            
        Returns:
            SVG points string: "x1,y1 x2,y2 ..."
        """
        if not polygon or len(polygon) < 4:
            return ""
        
        points = []
        for i in range(0, len(polygon), 2):
            if i + 1 >= len(polygon):
                break
                
            x = polygon[i]
            y = polygon[i + 1]
            
            # Use raw coordinates directly
            # No Y-flip needed as Azure is Top-Left origin
            points.append(f"{x},{y}")
            
        return " ".join(points)
    
    def _find_matching_box(
        self,
        field_value: str,
        page_number: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find layout box that matches the field value using text matching.
        
        Strategy:
        1. Look for exact match in line-level boxes first
        2. Fall back to word-level matches
        3. Filter by page if specified
        """
        if not field_value or not self.layout_data:
            return None
        
        field_value_clean = field_value.strip().lower()
        
        # Priority 1: Find matching lines (better for multi-word values)
        for box in self.layout_data:
            if box.get("type") == "line":
                content = (box.get("content") or "").strip().lower()
                if field_value_clean in content or content in field_value_clean:
                    if page_number is None or box.get("page_number") == page_number:
                        return box
        
        # Priority 2: Find matching words
        for box in self.layout_data:
            if box.get("type") == "word":
                content = (box.get("content") or "").strip().lower()
                if content == field_value_clean:
                    if page_number is None or box.get("page_number") == page_number:
                        return box
        
        return None
    
    def _generate_svg_overlay(
        self,
        page_num: int,
        key_bbox: Optional[Dict[str, Any]] = None,
        value_bbox: Optional[Dict[str, Any]] = None,
        show_all_boxes: bool = False
    ) -> str:
        """
        Generate SVG overlay with KEY (green) and VALUE (blue) bounding boxes.
        
        Args:
            page_num: Page number (1-indexed)
            key_bbox: Bounding box for field key/label (green)
            value_bbox: Bounding box for field value (blue)
            show_all_boxes: Whether to show all boxes faintly
            
        Returns:
            SVG element HTML string
        """
        page_key = str(page_num)
        dims = self.page_dimensions.get(page_key, {})
        page_width = dims.get("width_inches", dims.get("width", 8.5))
        page_height = dims.get("height_inches", dims.get("height", 11.0))
        
        svg_elements = []
        
        # Optionally draw all boxes faintly (gray)
        if show_all_boxes:
            for box in self.layout_data:
                if box.get("page_number") == page_num and box.get("type") in ["line", "word"]:
                    polygon = box.get("polygon", [])
                    points = self._azure_polygon_to_svg_points(polygon, page_width, page_height)
                    if points:
                        svg_elements.append(
                            f'<polygon points="{points}" '
                            f'fill="none" stroke="rgba(100,100,100,0.3)" stroke-width="0.3" />'
                        )
        
        # Draw KEY box in GREEN (Emerald)
        if key_bbox and key_bbox.get("polygon"):
            kb_page = key_bbox.get("page") or key_bbox.get("page_number", 1)
            if kb_page == page_num:
                polygon = key_bbox.get("polygon", [])
                points = self._azure_polygon_to_svg_points(polygon, page_width, page_height)
                if points:
                    svg_elements.append(f'''
                        <polygon points="{points}" 
                            fill="rgba(16, 185, 129, 0.2)" 
                            stroke="#10B981" 
                            stroke-width="2"
                            style="filter: drop-shadow(0 0 3px rgba(16, 185, 129, 0.4));" />
                    ''')
        
        # Draw VALUE box in BLUE (with animation)
        if value_bbox and value_bbox.get("polygon"):
            vb_page = value_bbox.get("page") or value_bbox.get("page_number", 1)
            if vb_page == page_num:
                polygon = value_bbox.get("polygon", [])
                points = self._azure_polygon_to_svg_points(polygon, page_width, page_height)
                if points:
                    svg_elements.append(f'''
                        <polygon points="{points}" 
                            fill="rgba(59, 130, 246, 0.25)" 
                            stroke="#3B82F6" 
                            stroke-width="2.5"
                            style="filter: drop-shadow(0 0 6px rgba(59, 130, 246, 0.5));">
                            <animate attributeName="stroke-opacity" values="1;0.5;1" dur="1.5s" repeatCount="indefinite"/>
                        </polygon>
                    ''')
        
        # Wrap in SVG with viewBox based on actual page dimensions
        # This aligns strict coordinates (inches/pixels) with the image
        svg = f'''
        <svg 
            viewBox="0 0 {page_width} {page_height}" 
            preserveAspectRatio="none"
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;">
            {" ".join(svg_elements)}
        </svg>
        '''
        
        return svg
    
    def render(
        self,
        selected_field_id: Optional[str] = None,
        selected_field_value: Optional[str] = None,
        selected_field_page: Optional[int] = None,
        key_bbox: Optional[Dict[str, Any]] = None,
        value_bbox: Optional[Dict[str, Any]] = None,
        show_all_boxes: bool = False
    ):
        """
        Render the interactive document viewer.
        
        Args:
            selected_field_id: Currently selected field ID
            selected_field_value: Value of selected field (for box matching)
            selected_field_page: Page number of selected field
            show_all_boxes: Whether to show all OCR boxes faintly
        """
        
        # Find matching bounding boxes for selected field
        # Use passed key_bbox and value_bbox if available, otherwise search layout_data
        active_key_bbox = key_bbox
        active_value_bbox = value_bbox
        target_page = 1
        
        if selected_field_value:
            # Fall back to layout search if value_bbox not provided
            if not active_value_bbox:
                legacy_box = self._find_matching_box(
                    selected_field_value,
                    page_number=selected_field_page
                )
                if legacy_box:
                    active_value_bbox = legacy_box
                    target_page = legacy_box.get("page_number", legacy_box.get("page", 1))
            elif active_value_bbox:
                target_page = active_value_bbox.get("page") or active_value_bbox.get("page_number", 1)
            elif selected_field_page:
                target_page = selected_field_page
        
        # Page navigation for multi-page documents
        # Page navigation state management
        page_key = f"viewer_page_{self.doc_id}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1
            
        # Auto-jump to page if a field is selected and we aren't on that page
        if selected_field_value and target_page != st.session_state[page_key]:
             st.session_state[page_key] = target_page
             
        current_page = st.session_state[page_key]
        
        # SYNC for KVEditor
        st.session_state["active_viewer_page"] = current_page
        
        # Fetch and display image with overlay
        image_b64 = self._get_image_as_base64(current_page)
        
        if not image_b64:
            st.warning(f"Processed image for page {current_page} not available. Showing original document.")
            # Fallback to original document viewer would go here
            return
        
        # Generate SVG overlay with dual KEY (green) + VALUE (blue) boxes
        svg_overlay = self._generate_svg_overlay(
            current_page,
            key_bbox=active_key_bbox,
            value_bbox=active_value_bbox,
            show_all_boxes=show_all_boxes
        )
        
        # Determine main box for zoom/focus on current page
        # Prioritize VALUE box, then KEY box
        page_highlighted_box = None
        
        if active_value_bbox:
            vb_page = active_value_bbox.get("page") or active_value_bbox.get("page_number", 1)
            if vb_page == current_page:
                page_highlighted_box = active_value_bbox
                
        if not page_highlighted_box and active_key_bbox:
            kb_page = active_key_bbox.get("page") or active_key_bbox.get("page_number", 1)
            if kb_page == current_page:
                page_highlighted_box = active_key_bbox
        
        # Calculate zoom and scroll for highlighted box
        zoom_style = ""
        if page_highlighted_box:
            # Slight zoom (1.1x) centered on the highlighted box
            polygon = page_highlighted_box.get("polygon", [])
            if len(polygon) >= 4:
                dims = self.page_dimensions.get(str(current_page), {})
                page_width = dims.get("width_inches", dims.get("width", 8.5))
                page_height = dims.get("height_inches", dims.get("height", 11.0))
                
                # Calculate center of box
                x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
                y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
                center_x_pct = (sum(x_coords) / len(x_coords) / page_width) * 100
                center_y_pct = (sum(y_coords) / len(y_coords) / page_height) * 100
                
                # Slight zoom transform centered on box
                zoom_style = f"transform: scale(1.05); transform-origin: {center_x_pct}% {center_y_pct}%; transition: transform 0.3s ease;"
        
        # Render image with SVG overlay using HTML
        html_content = f'''
        <div style="
            position: relative; 
            border-radius: 12px; 
            overflow: hidden; 
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            background: #f8fafc;
        ">
            <div style="position: relative; {zoom_style}">
                <img 
                    src="data:image/jpeg;base64,{image_b64}" 
                    style="width: 100%; height: auto; display: block;"
                    alt="Processed document page {current_page}"
                />
                {svg_overlay}
        </div>
        </div>
        '''
        
        # Add highlight info to the HTML content itself (not separate st.markdown)
        highlight_info = ""
        if page_highlighted_box and selected_field_value:
            truncated_value = selected_field_value[:50] + ('...' if len(selected_field_value or '') > 50 else '')
            page_info = f" (Page {current_page})" if self.total_pages > 1 else ""
            highlight_info = f'''
            <div style="
                background: linear-gradient(90deg, #3B82F6, #6366F1);
                color: white;
                padding: 8px 16px;
                border-radius: 8px;
                margin-bottom: 12px;
                font-size: 0.9rem;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            ">
                📍 Highlighting: <strong>{truncated_value}</strong>{page_info}
            </div>
            '''
        
        # Wrap everything in a complete HTML document for components.html
        full_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
            </style>
        </head>
        <body>
            {highlight_info}
            {html_content}
        </body>
        </html>
        '''
        
        # Use components.html for proper SVG rendering (st.markdown escapes SVG)
        # Use components.html for proper SVG rendering (st.markdown escapes SVG)
        components.html(full_html, height=650, scrolling=True)
        
        # Bottom Navigation Controls
        if self.total_pages > 1:
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1:
                if st.button("◀ Prev", key=f"prev_{self.doc_id}", disabled=current_page <= 1, use_container_width=True):
                    st.session_state[page_key] = max(1, current_page - 1)
                    st.rerun()
            with c2:
                st.markdown(
                    f"<div style='text-align: center; color: #64748b; font-weight: 500; padding-top: 8px;'>"
                    f"Page {current_page} of {self.total_pages}</div>", 
                    unsafe_allow_html=True
                )
            with c3:
                if st.button("Next ▶", key=f"next_{self.doc_id}", disabled=current_page >= self.total_pages, use_container_width=True):
                    st.session_state[page_key] = min(self.total_pages, current_page + 1)
                    st.rerun()


def render_interactive_viewer(
    doc_id: str,
    extraction_data: Dict[str, Any],
    selected_field: Optional[Dict[str, Any]] = None
):
    """
    Convenience function to render InteractiveDocumentViewer.
    
    Args:
        doc_id: Document UUID
        extraction_data: Full extraction response with layout_data
        selected_field: Currently selected field dict
    """
    import streamlit as st
    
    viewer = InteractiveDocumentViewer(
        doc_id=doc_id,
        layout_data=extraction_data.get("layout_data", []),
        page_dimensions=extraction_data.get("page_dimensions", {}),
        processed_image_paths=extraction_data.get("processed_image_paths", {}),
        total_pages=len(extraction_data.get("processed_image_paths", {})) or 1
    )
    
    # Read show_all_boxes from session state (set by KVEditor checkbox)
    show_all_boxes = st.session_state.get("show_all_boxes_toggle", False)
    
    # Extract key_bbox and value_bbox from selected field
    key_bbox = None
    value_bbox = None
    if selected_field:
        key_bbox = selected_field.get("key_bbox")
        value_bbox = selected_field.get("value_bbox")
    
    viewer.render(
        selected_field_id=selected_field.get("id") if selected_field else None,
        selected_field_value=selected_field.get("field_value") if selected_field else None,
        selected_field_page=selected_field.get("page_number") if selected_field else None,
        key_bbox=key_bbox,
        value_bbox=value_bbox,
        show_all_boxes=show_all_boxes
    )
