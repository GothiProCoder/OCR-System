"""
Working Document Viewer with Proper Page Navigation
Uses native Streamlit components that actually render correctly.
"""

import streamlit as st
import base64
from typing import Dict, Any, Optional, List
from services.api import APIClient


def render_working_viewer(
    doc_id: str,
    extraction_data: Dict[str, Any],
    selected_field: Optional[Dict[str, Any]] = None
):
    """
    Document viewer with working page navigation.
    Uses native Streamlit components (not custom HTML that breaks).
    """
    
    # Get data
    layout_data = extraction_data.get("layout_data", [])
    page_dimensions = extraction_data.get("page_dimensions", {})
    processed_image_paths = extraction_data.get("processed_image_paths", {})
    total_pages = len(processed_image_paths) if processed_image_paths else 1
    
    # === PAGE STATE ===
    page_key = f"viewer_page_{doc_id}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    
    # Auto-jump to page if field is selected
    if selected_field:
        bbox = selected_field.get("value_bbox") or selected_field.get("key_bbox")
        if bbox:
            target_page = bbox.get("page", bbox.get("page_number", 1))
        else:
            target_page = selected_field.get("page_number", 1)
        
        if target_page != st.session_state[page_key]:
            st.session_state[page_key] = target_page
    
    current_page = st.session_state[page_key]
    st.session_state["active_viewer_page"] = current_page
    
    # === PAGE NAVIGATION (VISIBLE!) ===
    st.markdown("""
        <style>
        .nav-container {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            padding: 12px 20px;
            border-radius: 12px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .page-title {
            color: white;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .page-info-text {
            color: #94A3B8;
            font-size: 0.85rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Navigation row
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])
    
    with nav_col1:
        if st.button("⏮️ First", key=f"first_{doc_id}", disabled=current_page <= 1, use_container_width=True):
            st.session_state[page_key] = 1
            st.rerun()
    
    with nav_col2:
        if st.button("◀ Prev", key=f"prev_{doc_id}", disabled=current_page <= 1, use_container_width=True):
            st.session_state[page_key] = max(1, current_page - 1)
            st.rerun()
    
    with nav_col3:
        st.markdown(f"""
            <div style="text-align: center; padding: 8px; background: #F1F5F9; border-radius: 8px;">
                <span style="font-weight: 700; color: #6366F1; font-size: 1.1rem;">Page {current_page}</span>
                <span style="color: #64748B;"> of {total_pages}</span>
            </div>
        """, unsafe_allow_html=True)
    
    with nav_col4:
        if st.button("Next ▶", key=f"next_{doc_id}", disabled=current_page >= total_pages, use_container_width=True):
            st.session_state[page_key] = min(total_pages, current_page + 1)
            st.rerun()
    
    with nav_col5:
        if st.button("Last ⏭️", key=f"last_{doc_id}", disabled=current_page >= total_pages, use_container_width=True):
            st.session_state[page_key] = total_pages
            st.rerun()
    
    # === GET IMAGE ===
    image_bytes = APIClient.get_processed_image_bytes(doc_id, current_page)
    
    if not image_bytes:
        st.error(f"Could not load image for page {current_page}")
        return
    
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Get page dimensions
    dims = page_dimensions.get(str(current_page), {})
    page_width = dims.get("width_inches", dims.get("width", 8.5))
    page_height = dims.get("height_inches", dims.get("height", 11.0))
    
    # === BUILD SVG OVERLAY ===
    svg_elements = []
    
    # Show all boxes toggle
    show_all = st.session_state.get("show_all_boxes_toggle", False)
    if show_all:
        for box in layout_data:
            if box.get("page_number") == current_page and box.get("type") in ["line", "word"]:
                polygon = box.get("polygon", [])
                if polygon and len(polygon) >= 4:
                    points = " ".join([f"{polygon[i]},{polygon[i+1]}" for i in range(0, len(polygon), 2)])
                    svg_elements.append(f'<polygon points="{points}" fill="none" stroke="rgba(100,100,100,0.3)" stroke-width="0.01"/>')
    
    # Selected field highlight
    if selected_field:
        for bbox_type, color, fill_opacity in [("key_bbox", "#10B981", 0.1), ("value_bbox", "#3B82F6", 0.15)]:
            bbox = selected_field.get(bbox_type)
            if bbox and bbox.get("polygon"):
                bbox_page = bbox.get("page") or bbox.get("page_number", 1)
                if bbox_page == current_page:
                    polygon = bbox["polygon"]
                    points = " ".join([f"{polygon[i]},{polygon[i+1]}" for i in range(0, len(polygon), 2)])
                    svg_elements.append(f'''
                        <polygon points="{points}" 
                            fill="rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {fill_opacity})"
                            stroke="{color}"
                            stroke-width="0.03">
                            <animate attributeName="stroke-opacity" values="1;0.5;1" dur="1.5s" repeatCount="indefinite"/>
                        </polygon>
                    ''')
    
    svg_content = "\n".join(svg_elements)
    
    # === RENDER DOCUMENT ===
    # Using st.components.html for the image + overlay (more reliable than custom HTML cards)
    viewer_html = f'''
    <div style="
        position: relative;
        background: #1E293B;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    ">
        <div style="position: relative; display: inline-block; width: 100%;">
            <img src="data:image/jpeg;base64,{image_b64}" 
                style="width: 100%; height: auto; border-radius: 8px; display: block;"/>
            <svg viewBox="0 0 {page_width} {page_height}" 
                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"
                preserveAspectRatio="none">
                {svg_content}
            </svg>
        </div>
    </div>
    '''
    
    import streamlit.components.v1 as components
    components.html(viewer_html, height=600, scrolling=False)
    
    # === VIEWER CONTROLS ===
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
    
    with ctrl_col1:
        st.checkbox("Show all bounding boxes", key="show_all_boxes_toggle")
    
    with ctrl_col2:
        if selected_field:
            field_name = selected_field.get("field_key", "Unknown")
            st.info(f"🎯 Focused: **{field_name}**")
    
    with ctrl_col3:
        zoom_level = st.select_slider(
            "Zoom",
            options=["75%", "100%", "125%", "150%"],
            value="100%",
            key=f"zoom_{doc_id}"
        )
