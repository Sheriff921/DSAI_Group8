"""
EDA Page
"""

import streamlit as st

from config import (
    EDA_IMAGES,
    POST_EDA_IMAGES,
)

from Utils.visualization import display_image

st.title("📊 Exploratory Data Analysis")

st.header("Preprocessing EDA")

for title, image in EDA_IMAGES.items():

    st.subheader(title)

    display_image(
        image,
        caption=title,
    )

st.divider()

st.header("Postprocessing EDA")

for title, image in POST_EDA_IMAGES.items():

    st.subheader(title)

    display_image(
        image,
        caption=title,
    )