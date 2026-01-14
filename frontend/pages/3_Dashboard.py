
import streamlit as st
import pandas as pd
import plotly.express as px
from services.api import APIClient
from components.core.metric_card import MetricCard
from utils.state import StateManager
from assets.icons import Icons

# --- Page Config ---
st.set_page_config(
    page_title="Dashboard | Lumina OCR",
    page_icon="⚡",
    layout="wide"
)

def render_metric_row(stats: dict):
    """Render the top row of KPI cards."""
    c1, c2, c3, c4 = st.columns(4)
    
    docs = stats.get("documents", {})
    extract = stats.get("extractions", {})
    proc = stats.get("processing", {})
    fields = stats.get("fields", {})

    with c1:
        MetricCard.display(
            label="Total Documents",
            value=str(docs.get("total", 0)),
            sub_value=f"+{docs.get('in_period', 0)} this week",
            trend="positive" if docs.get('in_period', 0) > 0 else "neutral"
        )
    
    with c2:
        success_rate = extract.get("success_rate", 0)
        MetricCard.display(
            label="Success Rate",
            value=f"{success_rate}%",
            sub_value="Extraction reliability",
            trend="positive" if success_rate > 90 else "negative"
        )

    with c3:
        MetricCard.display(
            label="Avg Confidence",
            value=f"{extract.get('avg_confidence', 0)}%",
            sub_value="AI Certainty",
            trend="neutral"
        )

    with c4:
        MetricCard.display(
            label="Processing Time",
            value=f"{proc.get('avg_total_time_ms', 0)}ms",
            sub_value="Avg per document",
            trend="neutral"
        )

def render_charts(stats: dict):
    """Render Plotly charts for analytics."""
    doc_stats = stats.get("documents", {})
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("### Document Status")
        status_data = doc_stats.get("by_status", {})
        if status_data:
            df_status = pd.DataFrame(list(status_data.items()), columns=["Status", "Count"])
            fig = px.pie(
                df_status, 
                values="Count", 
                names="Status", 
                hole=0.4,
                color_discrete_sequence=["#4F46E5", "#10B981", "#EF4444", "#F59E0B"] # Indigo, Emerald, Red, Amber
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No status data available.")

    with col2:
        st.markdown("### Document Types")
        type_data = doc_stats.get("by_form_type", {})
        if type_data:
            # Filter None/Unknown
            clean_data = {k: v for k, v in type_data.items() if k}
            if clean_data:
                df_types = pd.DataFrame(list(clean_data.items()), columns=["Type", "Count"])
                fig = px.bar(
                    df_types, 
                    x="Type", 
                    y="Count",
                    color="Count",
                    color_continuous_scale="indigo"
                )
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), xaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("No document types detected yet.")
        else:
            st.info("No type data available.")

def main():
    StateManager.init_state()
    
    # Header
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;">
            <div>
                <h1 style="font-size: 2.25rem; font-weight: 700; color: #0F172A; margin:0;">Analytics Dashboard</h1>
                <p style="color: #64748B; margin-top: 4px;">Overview of system performance and extraction metrics.</p>
            </div>
            <div style="background-color: #EFF6FF; padding: 8px 16px; border-radius: 8px; color: #1D4ED8; font-weight: 500; font-size: 0.9rem;">
                Last 7 Days
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Fetch Data
    stats = APIClient.get_dashboard_stats(period="week")
    
    if stats and not stats.get("error"):
        render_metric_row(stats)
        st.markdown("---")
        render_charts(stats)
        
        # System Health (Mini Footer)
        st.markdown("### System Status")
        st.markdown(
            """
            <div style="display: flex; gap: 12px; margin-top: 8px;">
                <span style="background-color: #ECFDF5; color: #047857; padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; display: flex; align-items: center; gap: 6px;">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #059669;"></div> Database Connected
                </span>
                <span style="background-color: #ECFDF5; color: #047857; padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; display: flex; align-items: center; gap: 6px;">
                     <div style="width: 8px; height: 8px; border-radius: 50%; background-color: #059669;"></div> Storage Active
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("Failed to load dashboard data. Check backend connection.")

if __name__ == "__main__":
    main()
