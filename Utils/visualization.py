"""
visualization.py

Visualization helpers used throughout the dashboard.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def display_image(image_path, caption=None):

    image_path = Path(image_path)

    if image_path.exists():

        st.image(
            str(image_path),
            caption=caption,
            use_container_width=True,
        )

    else:

        st.warning(
            f"Image not found:\n{image_path}"
        )


def comparison_table(df):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def prediction_bar_chart(df):

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    ax.bar(
        df["Model"],
        df["Prediction"],
    )

    ax.set_ylabel("Prediction")

    ax.set_title("Model Comparison")

    st.pyplot(fig)


def metric_card(title, value):

    st.metric(
        title,
        value,
    )