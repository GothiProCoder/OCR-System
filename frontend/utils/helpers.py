
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

def format_currency(value: float, currency: str = "USD") -> str:
    """Format float as currency string."""
    return f"${value:,.2f}"

def format_date(date_str: str) -> str:
    """Format ISO date string to readable format."""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime("%b %d, %Y • %I:%M %p")
    except:
        return date_str

def load_css(file_path: str):
    """Load and inject custom CSS."""
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def show_toast(message: str, type: str = "success"):
    """Show a styled toast notification."""
    if type == "success":
        st.toast(f"✅ {message}")
    elif type == "error":
        st.toast(f"❌ {message}")
    elif type == "warning":
        st.toast(f"⚠️ {message}")
    else:
        st.toast(message)
