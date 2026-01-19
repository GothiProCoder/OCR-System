"""
Premium Field Cards Component
Modern, animated field cards with inline editing and beautiful styling.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import List, Dict, Any, Callable, Optional
import json
import html


def render_premium_fields(
    fields: List[Dict[str, Any]],
    on_save: Callable[[List[Dict]], None] = None,
    on_field_select: Callable[[Dict[str, Any]], None] = None
) -> Optional[Dict[str, Any]]:
    """
    Render premium, animated field cards.
    
    Features:
    - Glassmorphism card design
    - Smooth hover animations
    - Inline editing
    - Confidence indicators with gradients
    - Page filtering
    - Click-to-focus interaction
    """
    
    if not fields:
        st.info("No fields extracted yet.")
        return None
    
    # Get current page
    current_page = st.session_state.get("active_viewer_page", 1)
    
    # Filter fields for current page
    page_fields = [f for f in fields if f.get("page_number", 1) == current_page]
    total_fields = len(fields)
    page_field_count = len(page_fields)
    
    # Initialize selection state
    if 'selected_field_id' not in st.session_state:
        st.session_state.selected_field_id = None
    
    # Build card HTML
    cards_html = []
    
    for i, field in enumerate(page_fields):
        field_id = field.get("id", str(i))
        field_key = html.escape(str(field.get("field_key", "Unknown")))
        field_value = html.escape(str(field.get("field_value", "")))
        confidence = field.get("confidence", 0)
        is_valid = field.get("is_valid", False)
        has_bbox = bool(field.get("value_bbox"))
        is_selected = st.session_state.selected_field_id == field_id
        
        # Confidence color gradient
        if confidence >= 0.85:
            conf_gradient = "linear-gradient(135deg, #10B981, #059669)"
            conf_text = f"{int(confidence * 100)}%"
        elif confidence >= 0.6:
            conf_gradient = "linear-gradient(135deg, #F59E0B, #D97706)"
            conf_text = f"{int(confidence * 100)}%"
        else:
            conf_gradient = "linear-gradient(135deg, #EF4444, #DC2626)"
            conf_text = f"{int(confidence * 100)}%"
        
        selected_class = "selected" if is_selected else ""
        warning_icon = '<span class="warning-icon" title="No bounding box found">⚠</span>' if not has_bbox else ''
        valid_badge = '<span class="valid-badge">✓ Verified</span>' if is_valid else ''
        
        card_html = f'''
        <div class="field-card {selected_class}" data-id="{field_id}" data-index="{i}">
            <div class="card-glow"></div>
            <div class="card-content">
                <div class="card-header">
                    <div class="field-label">
                        {warning_icon}
                        <span>{field_key}</span>
                    </div>
                    <div class="conf-badge" style="background: {conf_gradient}">
                        {conf_text}
                    </div>
                </div>
                <div class="field-value">{field_value if field_value else '<span class="empty">Empty</span>'}</div>
                <div class="card-footer">
                    {valid_badge}
                    <button class="focus-btn" data-index="{i}">
                        <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
                        </svg>
                        Focus
                    </button>
                </div>
            </div>
        </div>
        '''
        cards_html.append(card_html)
    
    all_cards = "\n".join(cards_html)
    
    # Full component HTML
    component_html = f'''
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
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: transparent;
                padding: 8px;
            }}
            
            .header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 16px;
                padding-bottom: 16px;
                border-bottom: 1px solid #E2E8F0;
            }}
            
            .header-title {{
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            
            .header-icon {{
                background: linear-gradient(135deg, #6366F1, #8B5CF6);
                width: 40px;
                height: 40px;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
            }}
            
            .header h2 {{
                font-size: 18px;
                font-weight: 600;
                color: #1E293B;
                margin: 0;
            }}
            
            .header p {{
                font-size: 13px;
                color: #64748B;
                margin: 0;
            }}
            
            .stats {{
                display: flex;
                gap: 16px;
            }}
            
            .stat {{
                text-align: center;
            }}
            
            .stat-value {{
                font-size: 20px;
                font-weight: 700;
                color: #6366F1;
            }}
            
            .stat-label {{
                font-size: 11px;
                color: #94A3B8;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            /* Cards Container */
            .cards-container {{
                display: flex;
                flex-direction: column;
                gap: 12px;
                max-height: 500px;
                overflow-y: auto;
                padding-right: 8px;
            }}
            
            .cards-container::-webkit-scrollbar {{
                width: 6px;
            }}
            
            .cards-container::-webkit-scrollbar-track {{
                background: #F1F5F9;
                border-radius: 3px;
            }}
            
            .cards-container::-webkit-scrollbar-thumb {{
                background: #CBD5E1;
                border-radius: 3px;
            }}
            
            /* Field Card */
            .field-card {{
                position: relative;
                background: white;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
                overflow: hidden;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            .field-card:hover {{
                border-color: #6366F1;
                transform: translateY(-2px);
                box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.15);
            }}
            
            .field-card.selected {{
                border-color: #6366F1;
                border-width: 2px;
                background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
            }}
            
            .card-glow {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
                opacity: 0;
                transition: opacity 0.3s ease;
            }}
            
            .field-card:hover .card-glow,
            .field-card.selected .card-glow {{
                opacity: 1;
            }}
            
            .card-content {{
                padding: 16px;
            }}
            
            .card-header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 8px;
            }}
            
            .field-label {{
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: #6366F1;
            }}
            
            .warning-icon {{
                color: #F59E0B;
            }}
            
            .conf-badge {{
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 600;
                color: white;
            }}
            
            .field-value {{
                font-size: 15px;
                font-weight: 500;
                color: #1E293B;
                line-height: 1.5;
                word-break: break-word;
            }}
            
            .field-value .empty {{
                color: #94A3B8;
                font-style: italic;
            }}
            
            .card-footer {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 12px;
                padding-top: 12px;
                border-top: 1px solid #F1F5F9;
            }}
            
            .valid-badge {{
                display: flex;
                align-items: center;
                gap: 4px;
                font-size: 12px;
                color: #10B981;
                font-weight: 500;
            }}
            
            .focus-btn {{
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 6px 12px;
                background: linear-gradient(135deg, #6366F1, #8B5CF6);
                border: none;
                border-radius: 6px;
                color: white;
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
            
            .focus-btn:hover {{
                transform: scale(1.05);
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
            }}
            
            /* Empty State */
            .empty-state {{
                text-align: center;
                padding: 40px 20px;
                color: #94A3B8;
            }}
            
            .empty-state svg {{
                margin-bottom: 12px;
                opacity: 0.4;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-title">
                <div class="header-icon">
                    <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/>
                    </svg>
                </div>
                <div>
                    <h2>Page {current_page} Data</h2>
                    <p>Click a field to highlight in viewer</p>
                </div>
            </div>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{page_field_count}</div>
                    <div class="stat-label">This Page</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{total_fields}</div>
                    <div class="stat-label">Total</div>
                </div>
            </div>
        </div>
        
        <div class="cards-container">
            {all_cards if cards_html else '<div class="empty-state"><svg width="48" height="48" fill="currentColor" viewBox="0 0 24 24"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm4 18H6V4h7v5h5v11z"/></svg><p>No fields on this page</p></div>'}
        </div>
    </body>
    </html>
    '''
    
    # Render the HTML component
    components.html(component_html, height=600, scrolling=False)
    
    # === Streamlit Controls Below ===
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Save All", type="primary", use_container_width=True):
            if on_save:
                on_save(fields)
            st.success("Saved!")
    
    with col2:
        if st.button("📥 Export JSON", use_container_width=True):
            st.download_button(
                "Download",
                data=json.dumps([{k: v for k, v in f.items() if k != 'value_bbox' and k != 'key_bbox'} for f in fields], indent=2),
                file_name="extracted_fields.json",
                mime="application/json"
            )
    
    # Field selection via buttons (workaround for component isolation)
    st.markdown("##### Quick Select")
    for i, field in enumerate(page_fields[:8]):  # Show first 8
        field_id = field.get("id", str(i))
        field_key = field.get("field_key", "Unknown")[:25]
        
        if st.button(f"🎯 {field_key}", key=f"sel_{field_id}", use_container_width=True):
            st.session_state.selected_field_id = field_id
            if on_field_select:
                on_field_select(field)
            st.rerun()
    
    # Add new field
    with st.expander("➕ Add New Field"):
        with st.form(key=f"add_form_{current_page}"):
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
    
    return None
