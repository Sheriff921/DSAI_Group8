"""
Main entry point for the Streamlit application.
"""

import streamlit as st

from config import (
    APP_ICON,
    APP_TITLE,
    LAYOUT,
    validate_project_structure,
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
)

missing_files = validate_project_structure()

if missing_files:

    st.error("Some required project files are missing.")

    for file in missing_files:
        st.write(f"- {file}")

    st.stop()

st.title(APP_TITLE)

st.markdown(
"""
## Welcome

This dashboard presents an AI-based energy consumption forecasting system
using three different prediction models:

- Prophet
- GRU
- CatBoost

Use the navigation menu on the left to explore:

- Home
- EDA
- Model Prediction
- Model Comparison
- About
"""
)

st.success("Project loaded successfully.")