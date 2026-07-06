"""
About Page
"""

import streamlit as st

st.title("ℹ️ About")

st.header("Project")

st.write("""
This project presents an AI-based energy consumption forecasting
system using three forecasting techniques:

- Prophet
- GRU
- CatBoost

The objective is to compare prediction performance across
multiple machine learning approaches using the World Bank
SE4ALL dataset.
""")

st.header("Dataset")

st.write("""
World Bank Sustainable Energy for All (SE4ALL)
""")

st.header("Methodology")

st.write("""
The workflow consists of:

- Data preprocessing
- Missing value handling
- Feature engineering
- Model training
- Performance evaluation
- Forecast comparison
""")

st.header("Authors")

st.write("""
DSAI Group 8
""")