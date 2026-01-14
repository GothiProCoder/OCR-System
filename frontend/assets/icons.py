
import textwrap

class Icons:
    """
    Central registry for application SVGs.
    All icons are optimized for 24x24px viewboxes unless specified.
    Colors use 'currentColor' where possible to inherit from parent text.
    """
    
    # --- Brand & Header ---
    
    @staticmethod
    def logo(color: str = "#4F46E5", size: int = 32) -> str:
        return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" fill="{color}" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

    # --- Navigation ---
    
    NAV_UPLOAD = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>"""
    
    NAV_HISTORY = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>"""
    
    NAV_DASHBOARD = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="9"/><rect x="14" y="3" width="7" height="5"/><rect x="14" y="12" width="7" height="9"/><rect x="3" y="16" width="7" height="5"/></svg>"""

    # --- Status Indicators ---
    
    @staticmethod
    def status_processing(color: str = "#4F46E5") -> str:
        return f"""<svg  width="20" height="20" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><style>.spinner_V8m1{{transform-origin:center;animation:spinner_zKoa 2s linear infinite}}.spinner_V8m1 circle{{stroke-linecap:round;animation:spinner_YpNB 1.5s ease-in-out infinite}}@keyframes spinner_zKoa{{100%{{transform:rotate(360deg)}}}}@keyframes spinner_YpNB{{0%{{stroke-dasharray:0 150;stroke-dashoffset:0}}47.5%{{stroke-dasharray:42 150;stroke-dashoffset:-16}}95%,100%{{stroke-dasharray:42 150;stroke-dashoffset:-59}}}}</style><g class="spinner_V8m1"><circle cx="12" cy="12" r="9.5" fill="none" stroke-width="3" stroke="{color}"></circle></g></svg>"""

    @staticmethod
    def status_completed(color: str = "#10B981") -> str:
        return f"""<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""
        
    @staticmethod
    def status_failed(color: str = "#EF4444") -> str:
        return f"""<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>"""
        
    @staticmethod
    def status_pending(color: str = "#64748B") -> str:
        return f"""<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>"""

    # --- Actions ---
    
    EDIT = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>"""
    
    DELETE = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>"""
    
    DOWNLOAD = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2 2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>"""

    @staticmethod
    def get_file_icon(mime_type: str) -> str:
        """Return appropriate icon for file type."""
        color = "#64748B"
        if "pdf" in mime_type:
            color = "#EF4444" # Red for PDF
        elif "image" in mime_type:
            color = "#4F46E5" # Indigo for Images
        elif "csv" in mime_type or "spreadsheet" in mime_type:
            color = "#10B981" # Green for Excel/CSV
            
        return f"""<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>"""
