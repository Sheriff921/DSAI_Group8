"""
model_loader.py

Loads all trained models and preprocessing artifacts.
"""

import pickle

import joblib
import streamlit as st
import torch
from catboost import CatBoostRegressor

from Utils.gru_model import GRUNet
from config import (
    CATBOOST_MODEL,
    ENCODER_FILE,
    GRU_MODEL,
    PROPHET_MODELS,
    PROPHET_SINGLE,
    SCALE_PARAM_FILE,
)


@st.cache_resource
def load_catboost():
    """Load trained CatBoost model."""

    model = CatBoostRegressor()
    model.load_model(str(CATBOOST_MODEL))
    return model


@st.cache_resource
def load_gru():
    """Load trained GRU model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUNet()

    model.load_state_dict(
        torch.load(
            GRU_MODEL,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    return model, device


@st.cache_resource
def load_prophet_models():
    """Load Prophet models."""

    with open(PROPHET_MODELS, "rb") as f:
        models = pickle.load(f)

    return models


@st.cache_resource
def load_single_prophet():
    """Load single-series Prophet."""

    return joblib.load(PROPHET_SINGLE)


@st.cache_resource
def load_encoders():

    with open(ENCODER_FILE, "rb") as f:
        encoders = pickle.load(f)

    return encoders


@st.cache_resource
def load_scale_parameters():

    with open(SCALE_PARAM_FILE, "rb") as f:
        params = pickle.load(f)

    return params


@st.cache_resource
def load_everything():
    """
    Convenience loader.
    """

    return {
        "catboost": load_catboost(),
        "gru": load_gru(),
        "prophet": load_prophet_models(),
        "prophet_single": load_single_prophet(),
        "encoders": load_encoders(),
        "scale_params": load_scale_parameters(),
    }