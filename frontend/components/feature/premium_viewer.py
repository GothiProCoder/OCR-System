"""
Premium Interactive Document Viewer
A truly modern, animated document viewer with smooth zoom and pan.
"""

import streamlit as st
import streamlit.components.v1 as components
import base64
from typing import List, Dict, Any, Optional
from services.api import APIClient


def render_premium_viewer(
    doc_id: str,
    extraction_data: Dict[str, Any],
    selected_field: Optional[Dict[str, Any]] = None
):
    """
    Render a premium, interactive document viewer with smooth animations.
    
    Features:
    - Smooth zoom-to-field animation
    - Page navigation with floating controls
    - Glassmorphism styling
    - Responsive SVG bounding box overlay
    - Dark/light theme support
    """
    
    # Extract data
    layout_data = extraction_data.get("layout_data", [])
    page_dimensions = extraction_data.get("page_dimensions", {})
    processed_image_paths = extraction_data.get("processed_image_paths", {})
    total_pages = len(processed_image_paths) or 1
    
    # Session state for current page
    page_key = f"viewer_page_{doc_id}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    
    # Determine target page from selected field
    target_page = 1
    if selected_field:
        bbox = selected_field.get("value_bbox") or selected_field.get("key_bbox")
        if bbox:
            target_page = bbox.get("page", 1)
        else:
            target_page = selected_field.get("page_number", 1)
        
        if target_page != st.session_state[page_key]:
            st.session_state[page_key] = target_page
    
    current_page = st.session_state[page_key]
    st.session_state["active_viewer_page"] = current_page
    
    # Get image for current page
    image_bytes = APIClient.get_processed_image_bytes(doc_id, current_page)
    if not image_bytes:
        st.warning("Image not available")
        return
    
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Get page dimensions
    dims = page_dimensions.get(str(current_page), {})
    page_width = dims.get("width_inches", dims.get("width", 8.5))
    page_height = dims.get("height_inches", dims.get("height", 11.0))
    
    # Build SVG overlay for bounding boxes
    svg_elements = []
    
    # Draw all boxes faintly
    show_all = st.session_state.get("show_all_boxes_toggle", False)
    if show_all:
        for box in layout_data:
            if box.get("page_number") == current_page and box.get("type") in ["line", "word"]:
                polygon = box.get("polygon", [])
                if polygon and len(polygon) >= 4:
                    points = " ".join([f"{polygon[i]},{polygon[i+1]}" for i in range(0, len(polygon), 2)])
                    svg_elements.append(f'<polygon points="{points}" fill="none" stroke="rgba(100,100,100,0.2)" stroke-width="0.02"/>')
    
    # Highlighted field boxes
    zoom_target = None
    if selected_field:
        key_bbox = selected_field.get("key_bbox")
        value_bbox = selected_field.get("value_bbox")
        
        # Key box (green)
        if key_bbox and key_bbox.get("polygon"):
            kb_page = key_bbox.get("page") or key_bbox.get("page_number", 1)
            if kb_page == current_page:
                polygon = key_bbox["polygon"]
                points = " ".join([f"{polygon[i]},{polygon[i+1]}" for i in range(0, len(polygon), 2)])
                svg_elements.append(f'''
                    <polygon points="{points}" 
                        fill="rgba(16, 185, 129, 0.15)" 
                        stroke="#10B981" 
                        stroke-width="0.03"
                        class="bbox-key"/>
                ''')
        
        # Value box (blue) - primary focus
        if value_bbox and value_bbox.get("polygon"):
            vb_page = value_bbox.get("page") or value_bbox.get("page_number", 1)
            if vb_page == current_page:
                polygon = value_bbox["polygon"]
                points = " ".join([f"{polygon[i]},{polygon[i+1]}" for i in range(0, len(polygon), 2)])
                svg_elements.append(f'''
                    <polygon points="{points}" 
                        fill="rgba(59, 130, 246, 0.2)" 
                        stroke="#3B82F6" 
                        stroke-width="0.04"
                        class="bbox-value animated"/>
                ''')
                # Calculate zoom target (center of box)
                x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
                y_coords = [polygon[i] for i in range(1, len(polygon), 2)]
                zoom_target = {
                    "x": sum(x_coords) / len(x_coords) / page_width * 100,
                    "y": sum(y_coords) / len(y_coords) / page_height * 100
                }
    
    svg_content = "\n".join(svg_elements)
    
    # Build the premium HTML viewer
    zoom_css = ""
    if zoom_target:
        zoom_css = f"--zoom-x: {zoom_target['x']}%; --zoom-y: {zoom_target['y']}%;"
    
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: transparent;
                overflow: hidden;
            }}
            
            .viewer-container {{
                display: flex;
                flex-direction: column;
                height: 100%;
                background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            }}
            
            /* Header with glassmorphism */
            .viewer-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 12px 20px;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .viewer-title {{
                display: flex;
                align-items: center;
                gap: 10px;
                color: white;
                font-size: 14px;
                font-weight: 500;
            }}
            
            .viewer-title svg {{
                opacity: 0.7;
            }}
            
            .page-indicator {{
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            
            .page-btn {{
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: white;
                padding: 8px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 13px;
                font-weight: 500;
                transition: all 0.2s ease;
            }}
            
            .page-btn:hover:not(:disabled) {{
                background: rgba(255, 255, 255, 0.2);
                transform: translateY(-1px);
            }}
            
            .page-btn:disabled {{
                opacity: 0.3;
                cursor: not-allowed;
            }}
            
            .page-info {{
                color: rgba(255, 255, 255, 0.7);
                font-size: 13px;
                min-width: 100px;
                text-align: center;
            }}
            
            /* Document canvas */
            .canvas-wrapper {{
                flex: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                overflow: hidden;
                position: relative;
            }}
            
            .document-canvas {{
                position: relative;
                max-width: 100%;
                max-height: 100%;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
                transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1);
                {zoom_css}
            }}
            
            .document-canvas.zoomed {{
                transform: scale(1.4);
                transform-origin: var(--zoom-x, 50%) var(--zoom-y, 50%);
            }}
            
            .document-canvas img {{
                display: block;
                max-width: 100%;
                max-height: 60vh;
                object-fit: contain;
            }}
            
            .bbox-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
            }}
            
            /* Bounding box animations */
            .bbox-value.animated {{
                animation: pulse 2s ease-in-out infinite;
            }}
            
            @keyframes pulse {{
                0%, 100% {{ 
                    stroke-opacity: 1;
                    filter: drop-shadow(0 0 8px rgba(59, 130, 246, 0.6));
                }}
                50% {{ 
                    stroke-opacity: 0.5;
                    filter: drop-shadow(0 0 16px rgba(59, 130, 246, 0.8));
                }}
            }}
            
            /* Highlight banner */
            .highlight-banner {{
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px 20px;
                background: linear-gradient(90deg, #6366F1, #8B5CF6);
                color: white;
                font-size: 13px;
            }}
            
            .highlight-banner svg {{
                flex-shrink: 0;
            }}
            
            .highlight-text {{
                font-weight: 500;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            
            /* Minimap */
            .minimap {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                background: rgba(0, 0, 0, 0.7);
                border-radius: 8px;
                padding: 8px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .minimap img {{
                width: 80px;
                height: auto;
                border-radius: 4px;
                opacity: 0.7;
            }}
            
            /* Controls overlay */
            .controls-overlay {{
                position: absolute;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                display: flex;
                gap: 8px;
            }}
            
            .control-btn {{
                background: rgba(0, 0, 0, 0.6);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            
            .control-btn:hover {{
                background: rgba(99, 102, 241, 0.8);
                transform: scale(1.1);
            }}
        </style>
    </head>
    <body>
        <div class="viewer-container">
            <!-- Header -->
            <div class="viewer-header">
                <div class="viewer-title">
                    <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm4 18H6V4h7v5h5v11z"/>
                    </svg>
                    <span>Document Viewer</span>
                </div>
                <div class="page-indicator">
                    <button class="page-btn" id="prevBtn" {"disabled" if current_page <= 1 else ""}>◀ Prev</button>
                    <span class="page-info">Page {current_page} of {total_pages}</span>
                    <button class="page-btn" id="nextBtn" {"disabled" if current_page >= total_pages else ""}>Next ▶</button>
                </div>
            </div>
            
            <!-- Highlight Banner -->
            {"<div class='highlight-banner'><svg width='16' height='16' fill='currentColor' viewBox='0 0 24 24'><path d='M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z'/></svg><span class='highlight-text'>Highlighting: " + (selected_field.get('field_value', '')[:60] if selected_field else '') + "</span></div>" if selected_field else ""}
            
            <!-- Canvas -->
            <div class="canvas-wrapper">
                <div class="document-canvas" id="docCanvas" {"class='zoomed'" if zoom_target else ""}>
                    <img src="data:image/jpeg;base64,{image_b64}" alt="Document" />
                    <svg class="bbox-overlay" viewBox="0 0 {page_width} {page_height}" preserveAspectRatio="none">
                        {svg_content}
                    </svg>
                </div>
            </div>
        </div>
        
        <script>
            // Page navigation
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            
            if (prevBtn) {{
                prevBtn.addEventListener('click', () => {{
                    window.parent.postMessage({{type: 'page_change', page: {current_page - 1}}}, '*');
                }});
            }}
            
            if (nextBtn) {{
                nextBtn.addEventListener('click', () => {{
                    window.parent.postMessage({{type: 'page_change', page: {current_page + 1}}}, '*');
                }});
            }}
            
            // Auto-zoom on load if target exists
            const canvas = document.getElementById('docCanvas');
            if (canvas && canvas.classList.contains('zoomed')) {{
                setTimeout(() => {{
                    canvas.classList.add('zoomed');
                }}, 100);
            }}
        </script>
    </body>
    </html>
    '''
    
    # Render the component
    components.html(html, height=700, scrolling=False)
    
    # Handle page navigation via session state workaround
    # (Since postMessage won't directly update session state)
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("◀ Previous", key=f"prev_{doc_id}", disabled=current_page <= 1):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with nav_col2:
        st.checkbox("Show all boxes", key="show_all_boxes_toggle")
    with nav_col3:
        if st.button("Next ▶", key=f"next_{doc_id}", disabled=current_page >= total_pages):
            st.session_state[page_key] = current_page + 1
            st.rerun()
