
import streamlit as st

class MetricCard:
    """
    Renders a statistics card with label, value, and optional trend indicator.
    Uses generic styling classes from global CSS.
    """
    
    @staticmethod
    def _trend_icon(is_positive: bool) -> str:
        color = "#10B981" if is_positive else "#EF4444"
        path = "M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" if is_positive else "M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"
        
        return f"""
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="{path}"/>
        </svg>
        """

    @staticmethod
    def display(label: str, value: str, sub_value: str = None, trend: str = "neutral"):
        """
        Render a metric card.
        trend: 'up', 'down', or 'neutral'
        """
        
        trend_html = ""
        if sub_value:
            icon = ""
            color = "#64748B"
            
            if trend == "up":
                icon = MetricCard._trend_icon(True)
                color = "#10B981"
            elif trend == "down":
                icon = MetricCard._trend_icon(False)
                color = "#EF4444"
                
            trend_html = f"""
            <div style="display: flex; align-items: center; gap: 4px; margin-top: 8px; font-size: 0.85rem; color: {color};">
                {icon}
                <span>{sub_value}</span>
            </div>
            """
            
        st.markdown(
            f"""
            <div class="lumina-card" style="padding: 1.25rem;">
                <div style="color: #64748B; font-size: 0.875rem; font-weight: 500; margin-bottom: 4px;">
                    {label}
                </div>
                <div style="font-size: 1.875rem; font-weight: 700; color: #0F172A; letter-spacing: -0.025em; line-height: 1.2;">
                    {value}
                </div>
                {trend_html}
            </div>
            """,
            unsafe_allow_html=True
        )
