"""
Home Page
"""

import streamlit as st

st.title("⚡ AI-based Energy Consumption Forecasting")

st.markdown("""
Welcome to the AI-Based Energy Consumption Forecasting Dashboard.

This project compares three machine learning and time-series forecasting
models trained using the World Bank SE4ALL dataset.

### Models Included

- Prophet
- GRU (Gated Recurrent Unit)
- CatBoost

---

### Workflow

1. Raw Dataset
2. Data Cleaning
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Prediction Comparison

---

### Navigation

Use the sidebar to explore:

- 📊 Exploratory Data Analysis
- 🤖 Model Prediction
- 📈 Model Comparison
- ℹ️ About
""")

col1, col2, col3 = st.columns(3)

col1.metric("Models", "3")
col2.metric("Dataset", "World Bank SE4ALL")
col3.metric("Prediction Type", "Time Series")

st.success("Dashboard Ready")