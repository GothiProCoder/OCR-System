
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Callable
from assets.icons import Icons
from services.api import APIClient

class KVEditor:
    """
    Interactive Key-Value Editor for Extracted Data.
    Allows users to view, edit, and validate fields.
    """

    @staticmethod
    def render(fields: List[Dict[str, Any]], on_save: Callable[[List[Dict]], None] = None):
        """
        Render the data editor.
        fields: List ofExtractedField objects (dicts).
        on_save: Callback when changes are saved.
        """
        if not fields:
            st.info("No fields extracted yet.")
            return

        # 1. Prepare Data for Editor
        # We flatten the structure slightly for the DataFrame
        df = pd.DataFrame(fields)
        
        # Ensure columns exist even if empty
        required_cols = ["id", "field_key", "field_value", "field_type", "confidence", "is_valid"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        # Select display columns (id is hidden but tracked)
        # We rename columns for better UI display
        display_df = df[["id", "field_key", "field_value", "field_type", "confidence", "is_valid"]].copy()
        
        # 2. Render Header with Actions
        c1, c2 = st.columns([6, 4])
        with c1:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <div style="color: #4F46E5;">{Icons.EDIT.replace('width="16"', 'width="24"').replace('height="16"', 'height="24"')}</div>
                    <h3 style="margin: 0;">Extraction Results</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with c2:
            # Action Bar (Save / Download)
            # Using columns for buttons to align right
            b1, b2 = st.columns([1, 1])
            pass

        # 3. Data Editor (The Core)
        edited_df = st.data_editor(
            display_df,
            column_config={
                "id": None, # Hide ID
                "field_key": st.column_config.TextColumn("Field Label", disabled=False),
                "field_value": st.column_config.TextColumn("Extracted Value", disabled=False),
                "field_type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["text", "date", "number", "email", "phone", "name", "address", "currency", "table"],
                    width="small"
                ),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    help="AI Confidence Score",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
                "is_valid": st.column_config.CheckboxColumn("Valid?", default=True)
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic", # Allow adding/deleting rows
            key="kv_editor_grid"
        )
        
        # 4. Save Logic
        # Compare Edited DF with Original DF to detect changes
        # (Simplified: User clicks button to commit)
        
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        
        col_save, col_dl = st.columns([1, 4])
        
        with col_save:
            if st.button("Save Changes", type="primary", use_container_width=True):
                # Convert back to list of dicts
                updated_data = edited_df.to_dict(orient="records")
                if on_save:
                    on_save(updated_data) # Trigger callback
                else:
                    st.success("Changes saved locally.")
        
        with col_dl:
            # Download JSON
            json_str = edited_df.to_json(orient="records", indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="extraction_results.json",
                mime="application/json",
                icon=None # No emoji, use text
            )

