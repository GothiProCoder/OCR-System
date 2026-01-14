
import streamlit as st
from assets.icons import Icons

class StatusBadge:
    """
    Renders status indicators using SVG icons instead of emojis.
    Follows Lumina Design System color conventions.
    """
    
    @staticmethod
    def _render_badge(icon_svg: str, label: str, bg_color: str, text_color: str):
        """Internal helper to render the badge HTML."""
        st.markdown(
            f"""<div style="display: inline-flex; align-items: center; gap: 6px; background-color: {bg_color}; color: {text_color}; padding: 4px 10px; border-radius: 9999px; font-size: 0.85rem; font-weight: 500; border: 1px solid {text_color}20;">
                <span style="display: flex; align-items: center; width: 16px;">{icon_svg}</span>
                <span>{label}</span>
            </div>""",
            unsafe_allow_html=True
        )

    @staticmethod
    def processing():
        StatusBadge._render_badge(
            icon_svg=Icons.status_processing(color="#4F46E5"),
            label="Processing",
            bg_color="#EEF2FF", # Indigo-50
            text_color="#4338CA" # Indigo-700
        )

    @staticmethod
    def completed():
        StatusBadge._render_badge(
            icon_svg=Icons.status_completed(color="#059669"),
            label="Completed",
            bg_color="#ECFDF5", # Emerald-50
            text_color="#047857" # Emerald-700
        )

    @staticmethod
    def failed(error_msg: str = None):
        StatusBadge._render_badge(
            icon_svg=Icons.status_failed(color="#DC2626"),
            label="Failed",
            bg_color="#FEF2F2", # Red-50
            text_color="#B91C1C" # Red-700
        )
        if error_msg:
            st.markdown(
                f'<div style="font-size: 0.75rem; color: #EF4444; margin-top: 4px;">{error_msg}</div>', 
                unsafe_allow_html=True
            )

    @staticmethod
    def pending():
        StatusBadge._render_badge(
            icon_svg=Icons.status_pending(color="#475569"),
            label="Pending",
            bg_color="#F1F5F9", # Slate-100
            text_color="#334155" # Slate-700
        )
        
    @staticmethod
    def custom(label: str, color: str = "#64748B"):
        """Generic badge for other statuses."""
        # Simple dot icon for generic states
        dot_svg = f"""
        <svg width="16" height="16" viewBox="0 0 24 24" fill="{color}">
            <circle cx="12" cy="12" r="6"/>
        </svg>
        """
        StatusBadge._render_badge(
            icon_svg=dot_svg,
            label=label,
            bg_color="#F8FAFC",
            text_color=color
        )
