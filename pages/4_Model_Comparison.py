"""
Model Comparison Page
"""

import streamlit as st

from config import (
    MODEL_COMPARISON_IMAGE,
)

from Utils.visualization import (
    display_image,
)

st.title("📈 Model Comparison")

st.markdown("""
Comparison of Prophet, GRU and CatBoost using the evaluation
performed during model development.
""")

display_image(
    MODEL_COMPARISON_IMAGE,
    caption="Model Comparison",
)