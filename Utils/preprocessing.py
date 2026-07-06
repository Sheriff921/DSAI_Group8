"""
preprocessing.py

Loads processed dataset and provides helper functions used
throughout the Streamlit application.
"""

import pandas as pd
import streamlit as st

from config import (
    GROUP_COLUMNS,
    PROCESSED_DATASET,
)


@st.cache_data
def load_processed_dataset():
    """
    Load processed dataset.
    """

    return pd.read_csv(PROCESSED_DATASET)


def get_available_countries(df):

    return sorted(df["COUNTRY_ID"].unique())


def get_available_indicators(df, country):

    subset = df[df["COUNTRY_ID"] == country]

    return sorted(subset["INDICATOR_ID"].unique())


def get_available_components(df, country, indicator):

    subset = df[
        (df["COUNTRY_ID"] == country)
        &
        (df["INDICATOR_ID"] == indicator)
    ]

    return sorted(subset["COMP_ID"].unique())


def filter_series(
    df,
    country,
    indicator,
    component,
):
    """
    Return one complete time series.
    """

    return (
        df[
            (df["COUNTRY_ID"] == country)
            &
            (df["INDICATOR_ID"] == indicator)
            &
            (df["COMP_ID"] == component)
        ]
        .sort_values("TIME_PERIOD")
        .reset_index(drop=True)
    )


def get_all_series(df):
    """
    Group by series.
    """

    return df.groupby(GROUP_COLUMNS)