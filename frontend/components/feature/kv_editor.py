
import streamlit as st
import streamlit.components.v1 as components
from typing import List, Dict, Any, Callable, Optional
import json

class KVEditor:
    """
    Professional Key-Value Editor with modern card design.
    Features: Page filtering, inline editing, validation, and viewer sync.
    """

    @staticmethod
    def render(
        fields: List[Dict[str, Any]], 
        on_save: Callable[[List[Dict]], None] = None,
        on_field_select: Callable[[Dict[str, Any]], None] = None
    ) -> Optional[Dict[str, Any]]:
        """Render professional field cards with page filtering."""
        
        if not fields:
            st.info("No fields extracted yet.")
            return None

        # Get current page from viewer sync
        current_page = st.session_state.get("active_viewer_page", 1)
        
        # Filter fields for current page
        page_fields = [f for f in fields if f.get("page_number", 1) == current_page]
        total_pages = max([f.get("page_number", 1) for f in fields], default=1)
        
        # Initialize selected field tracking
        if 'kv_selected_id' not in st.session_state:
            st.session_state.kv_selected_id = None

        # === HEADER ===
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <div style="
                        background: linear-gradient(135deg, #6366F1, #8B5CF6);
                        padding: 8px;
                        border-radius: 10px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">
                        <svg width="20" height="20" fill="white" viewBox="0 0 24 24">
                            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
                        </svg>
                    </div>
                    <div>
                        <h3 style="margin: 0; font-size: 1.1rem; color: #1E293B;">Page {current_page} of {total_pages}</h3>
                        <p style="margin: 0; font-size: 0.8rem; color: #64748B;">{len(page_fields)} fields on this page</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with header_col2:
            if st.button("💾 Save All", type="primary", use_container_width=True):
                if on_save:
                    on_save(fields)
                st.success("Saved!")

        # === INJECT CSS FOR PROFESSIONAL CARDS ===
        st.markdown("""
        <style>
        /* Card Container */
        .field-card {
            background: linear-gradient(145deg, #FFFFFF, #F8FAFC);
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            transition: all 0.2s ease;
            cursor: pointer;
            position: relative;
        }
        .field-card:hover {
            border-color: #6366F1;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
            transform: translateY(-2px);
        }
        .field-card.selected {
            border-color: #6366F1;
            border-width: 2px;
            background: linear-gradient(145deg, #EEF2FF, #E0E7FF);
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);
        }
        
        /* Field Label */
        .field-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #6366F1;
            margin-bottom: 4px;
        }
        
        /* Field Value */
        .field-value {
            font-size: 1rem;
            color: #1E293B;
            font-weight: 500;
            line-height: 1.4;
            word-break: break-word;
        }
        
        /* Confidence Badge */
        .conf-badge {
            position: absolute;
            top: 12px;
            right: 12px;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        .conf-high { background: #DCFCE7; color: #166534; }
        .conf-mid { background: #FEF3C7; color: #92400E; }
        .conf-low { background: #FEE2E2; color: #991B1B; }
        
        /* Valid Indicator */
        .valid-check {
            position: absolute;
            bottom: 12px;
            right: 12px;
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 0.75rem;
            color: #16A34A;
        }
        
        /* Warning Badge */
        .warning-badge {
            position: absolute;
            top: 12px;
            left: 12px;
            color: #EAB308;
            font-size: 0.9rem;
        }
        
        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #94A3B8;
        }
        .empty-state svg {
            margin-bottom: 12px;
            opacity: 0.5;
        }
        </style>
        """, unsafe_allow_html=True)

        # === RENDER FIELD CARDS ===
        if not page_fields:
            st.markdown("""
                <div class="empty-state">
                    <svg width="48" height="48" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm4 18H6V4h7v5h5v11z"/>
                    </svg>
                    <p>No fields found on this page</p>
                </div>
            """, unsafe_allow_html=True)
            return None

        selected_field = None
        
        for i, field in enumerate(page_fields):
            field_id = field.get("id", str(i))
            field_key = field.get("field_key", "Unknown Field")
            field_value = field.get("field_value", "")
            confidence = field.get("confidence", 0)
            is_valid = field.get("is_valid", False)
            has_bbox = bool(field.get("value_bbox"))
            
            # Confidence styling
            if confidence >= 0.85:
                conf_class = "conf-high"
                conf_text = f"✓ {int(confidence * 100)}%"
            elif confidence >= 0.6:
                conf_class = "conf-mid"
                conf_text = f"~ {int(confidence * 100)}%"
            else:
                conf_class = "conf-low"
                conf_text = f"! {int(confidence * 100)}%"
            
            # Check if selected
            is_selected = st.session_state.kv_selected_id == field_id
            card_class = "field-card selected" if is_selected else "field-card"
            
            # Warning for no bbox
            warning_html = '<span class="warning-badge" title="No bounding box found">⚠</span>' if not has_bbox else ''
            
            # Valid indicator
            valid_html = '<span class="valid-check">✓ Valid</span>' if is_valid else ''
            
            # Render card as button for selection
            col1, col2 = st.columns([6, 1])
            
            with col1:
                # Use button styled as card
                if st.button(
                    f"📍 {field_key}: {field_value[:60]}{'...' if len(str(field_value)) > 60 else ''}" if field_value else f"📍 {field_key}",
                    key=f"card_{field_id}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state.kv_selected_id = field_id
                    if on_field_select:
                        on_field_select(field)
                    st.rerun()
            
            with col2:
                # Confidence badge
                st.markdown(f'<span class="conf-badge {conf_class}">{conf_text}</span>', unsafe_allow_html=True)
            
            # Track selected
            if is_selected:
                selected_field = field
                
                # Show edit form for selected field
                with st.container():
                    st.markdown("---")
                    edit_col1, edit_col2 = st.columns([4, 1])
                    with edit_col1:
                        new_val = st.text_input(
                            f"Edit: {field_key}",
                            value=field_value,
                            key=f"edit_{field_id}"
                        )
                        field["field_value"] = new_val
                    with edit_col2:
                        is_valid_new = st.checkbox("Valid", value=is_valid, key=f"valid_{field_id}")
                        field["is_valid"] = is_valid_new
                    st.markdown("---")

        # === ADD NEW FIELD ===
        with st.expander("➕ Add New Field"):
            with st.form(key=f"add_field_{current_page}"):
                new_key = st.text_input("Field Name")
                new_val = st.text_input("Value")
                if st.form_submit_button("Add"):
                    import uuid
                    new_field = {
                        "id": str(uuid.uuid4()),
                        "field_key": new_key,
                        "field_value": new_val,
                        "confidence": 1.0,
                        "page_number": current_page,
                        "is_valid": True,
                        "field_type": "text"
                    }
                    fields.append(new_field)
                    if on_save:
                        on_save(fields)
                    st.rerun()
                    
        return selected_field
