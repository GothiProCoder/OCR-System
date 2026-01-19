"""
Interactive Field Editor with Dropdown Behavior
Shows only keys initially, expands to edit on click.
"""

import streamlit as st
from typing import List, Dict, Any, Callable, Optional
import uuid


def render_dropdown_fields(
    fields: List[Dict[str, Any]],
    on_save: Callable[[List[Dict]], None] = None,
    on_field_select: Callable[[Dict[str, Any]], None] = None
) -> Optional[Dict[str, Any]]:
    """
    Render fields with accordion/dropdown behavior.
    - Shows only KEY initially
    - On click: expands to show value editor + highlights bbox
    """
    
    if not fields:
        st.info("No fields extracted yet.")
        return None
    
    # Get current page from viewer
    current_page = st.session_state.get("active_viewer_page", 1)
    
    # Filter fields for current page
    page_fields = [f for f in fields if f.get("page_number", 1) == current_page]
    total_pages = max([f.get("page_number", 1) for f in fields], default=1)
    total_fields = len(fields)
    
    # Track which field is expanded
    if 'expanded_field_id' not in st.session_state:
        st.session_state.expanded_field_id = None

    # === HEADER WITH PAGE INFO ===
    st.markdown("""
        <style>
        .field-header {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
            padding: 16px 20px;
            border-radius: 12px;
            margin-bottom: 16px;
            color: white;
        }
        .field-header h3 {
            margin: 0;
            font-size: 1.1rem;
        }
        .field-header p {
            margin: 4px 0 0 0;
            opacity: 0.8;
            font-size: 0.85rem;
        }
        .field-stats {
            display: flex;
            gap: 20px;
            margin-top: 12px;
        }
        .stat-box {
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 8px;
        }
        .stat-box strong {
            font-size: 1.2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="field-header">
            <h3>📄 Page {current_page} of {total_pages}</h3>
            <p>Click on a field to edit and highlight</p>
            <div class="field-stats">
                <div class="stat-box"><strong>{len(page_fields)}</strong> fields on this page</div>
                <div class="stat-box"><strong>{total_fields}</strong> total fields</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # === DROPDOWN FIELD CARDS ===
    st.markdown("""
        <style>
        div[data-testid="stExpander"] {
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            margin-bottom: 8px;
            overflow: hidden;
        }
        div[data-testid="stExpander"]:hover {
            border-color: #6366F1;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
        }
        div[data-testid="stExpander"] summary {
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if not page_fields:
        st.warning("No fields found on this page. Try navigating to another page.")
        return None
    
    # Render each field as an expander (accordion behavior)
    for i, field in enumerate(page_fields):
        field_id = field.get("id", str(i))
        field_key = field.get("field_key", "Unknown")
        field_value = field.get("field_value", "")
        confidence = field.get("confidence", 0)
        is_valid = field.get("is_valid", False)
        has_bbox = bool(field.get("value_bbox") or field.get("key_bbox"))
        
        # Confidence indicator
        if confidence >= 0.85:
            conf_color = "🟢"
            conf_label = "High"
        elif confidence >= 0.6:
            conf_color = "🟡"
            conf_label = "Medium"
        else:
            conf_color = "🔴"
            conf_label = "Low"
        
        # Warning for missing bbox
        bbox_indicator = "" if has_bbox else " ⚠️"
        
        # Expander title shows key + confidence
        expander_title = f"{field_key}{bbox_indicator} — {conf_color} {int(confidence * 100)}%"
        
        with st.expander(expander_title, expanded=(st.session_state.expanded_field_id == field_id)):
            # When expanded, trigger field selection for bbox highlight
            if st.session_state.expanded_field_id != field_id:
                # First time opening this expander
                pass
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Editable value
                new_value = st.text_area(
                    "Value",
                    value=field_value,
                    key=f"val_{field_id}",
                    height=80,
                    label_visibility="collapsed"
                )
                field["field_value"] = new_value
            
            with col2:
                # Valid checkbox
                is_valid_new = st.checkbox(
                    "✓ Valid",
                    value=is_valid,
                    key=f"valid_{field_id}"
                )
                field["is_valid"] = is_valid_new
                
                # Focus button to highlight bbox
                if st.button("🎯 Focus", key=f"focus_{field_id}", use_container_width=True):
                    st.session_state.expanded_field_id = field_id
                    if on_field_select:
                        on_field_select(field)
                    st.rerun()
            
            # Confidence bar
            st.progress(confidence, text=f"Confidence: {conf_label} ({int(confidence*100)}%)")
            
            # Metadata
            st.caption(f"ID: `{field_id[:8]}...` | Page: {field.get('page_number', 1)} | Type: {field.get('field_type', 'text')}")

    # === ACTION BUTTONS ===
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save All Changes", type="primary", use_container_width=True):
            if on_save:
                on_save(fields)
            st.success("✓ All changes saved!")
    
    with col2:
        # Export as JSON
        import json
        export_data = [{k: v for k, v in f.items() if k not in ['value_bbox', 'key_bbox']} for f in fields]
        st.download_button(
            "📥 Export JSON",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name="extracted_fields.json",
            mime="application/json",
            use_container_width=True
        )

    # === ADD NEW FIELD ===
    st.markdown("---")
    st.markdown("### ➕ Add New Field")
    
    with st.form(key=f"add_field_form_{current_page}", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            new_key = st.text_input("Field Name", placeholder="e.g., Phone Number")
        with col2:
            new_value = st.text_input("Value", placeholder="e.g., +1-555-1234")
        
        col3, col4 = st.columns(2)
        with col3:
            new_type = st.selectbox("Field Type", ["text", "number", "date", "email", "phone", "address"])
        with col4:
            new_valid = st.checkbox("Mark as Valid", value=True)
        
        submit = st.form_submit_button("Add Field", type="primary", use_container_width=True)
        
        if submit and new_key:
            new_field = {
                "id": str(uuid.uuid4()),
                "field_key": new_key,
                "field_value": new_value,
                "confidence": 1.0,
                "page_number": current_page,
                "is_valid": new_valid,
                "field_type": new_type
            }
            fields.append(new_field)
            st.success(f"✓ Added field: {new_key}")
            if on_save:
                on_save(fields)
            st.rerun()
    
    return None
