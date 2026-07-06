"""
prediction.py

Inference utilities for Prophet, GRU and CatBoost.
"""

import numpy as np
import pandas as pd
import torch

from Utils.feature_engineering import (
    build_catboost_features,
)

from config import (
    GROUP_COLUMNS,
    TARGET_COLUMN,
    SEQUENCE_LENGTH,
)


def predict_catboost(model, series_df, feature_columns):
    """
    Predict using CatBoost.
    """

    feature_df = build_catboost_features(series_df)

    if feature_df.empty:
        return None

    X = feature_df[feature_columns].values

    prediction = model.predict(X)

    return float(prediction[-1])


def predict_gru(model, device, series_df):
    """
    Predict using trained GRU model.
    """

    sequence = (
        series_df["OBS_VALUE_SCALED"]
        .values
        .astype(np.float32)
    )

    if len(sequence) < SEQUENCE_LENGTH:
        return None

    sequence = sequence[-SEQUENCE_LENGTH:]

    sequence = (
        torch.tensor(sequence)
        .view(1, SEQUENCE_LENGTH, 1)
        .to(device)
    )

    model.eval()

    with torch.no_grad():

        prediction = model(sequence)

    return float(prediction.cpu().numpy()[0][0])


def predict_prophet(
    prophet_models,
    country,
    indicator,
    component,
):
    """
    Predict using fitted Prophet model.
    """

    key = (
        country,
        indicator,
        component,
    )

    if key not in prophet_models:
        return None

    model = prophet_models[key]

    future = model.make_future_dataframe(
        periods=1,
        freq="YS",
    )

    forecast = model.predict(future)

    return float(
        forecast["yhat"].iloc[-1]
    )


def compare_models(
    prophet_prediction,
    gru_prediction,
    catboost_prediction,
):
    """
    Return comparison table.
    """

    return pd.DataFrame(
        {
            "Model": [
                "Prophet",
                "GRU",
                "CatBoost",
            ],
            "Prediction": [
                prophet_prediction,
                gru_prediction,
                catboost_prediction,
            ],
        }
    )