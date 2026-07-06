"""
Prediction Page
"""

import streamlit as st

from Utils.model_loader import (
    load_everything,
)

from Utils.preprocessing import (
    filter_series,
    get_available_components,
    get_available_countries,
    get_available_indicators,
    load_processed_dataset,
)

from Utils.prediction import (
    compare_models,
    predict_catboost,
    predict_gru,
    predict_prophet,
)

from Utils.visualization import (
    comparison_table,
    prediction_bar_chart,
)

st.title("🤖 Model Prediction")

models = load_everything()

df = load_processed_dataset()

country = st.selectbox(
    "Country",
    get_available_countries(df),
)

indicator = st.selectbox(
    "Indicator",
    get_available_indicators(
        df,
        country,
    ),
)

component = st.selectbox(
    "Component",
    get_available_components(
        df,
        country,
        indicator,
    ),
)

if st.button("Run Prediction"):

    series = filter_series(
        df,
        country,
        indicator,
        component,
    )

    feature_columns = [
        "COUNTRY_ID",
        "INDICATOR_ID",
        "COMP_ID",
        "TIME_PERIOD",
        "OBS_VALUE",
        "DECADE",
        "LAG_1",
        "LAG_2",
        "LAG_3",
        "ROLLING_MEAN_3",
        "ROLLING_STD_3",
        "YOY_DIFF",
    ]

    prophet = predict_prophet(
        models["prophet"],
        country,
        indicator,
        component,
    )

    gru_model, device = models["gru"]

    gru = predict_gru(
        gru_model,
        device,
        series,
    )

    catboost = predict_catboost(
        models["catboost"],
        series,
        feature_columns,
    )

    results = compare_models(
        prophet,
        gru,
        catboost,
    )

    st.subheader("Prediction Results")

    comparison_table(results)

    prediction_bar_chart(results)