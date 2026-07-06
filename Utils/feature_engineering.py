"""
feature_engineering.py

Feature engineering functions used by CatBoost and GRU.
"""

import numpy as np

from config import (
    GROUP_COLUMNS,
    LAG_FEATURES,
    ROLLING_MEAN_COLUMN,
    ROLLING_STD_COLUMN,
    SEQUENCE_LENGTH,
    TARGET_COLUMN,
    TIME_COLUMN,
    YOY_COLUMN,
)


def build_gru_sequences(df):
    """
    Build GRU sequence windows.
    """

    groups = df.groupby(GROUP_COLUMNS)

    all_x = []
    all_y = []
    keys = []

    for key, group in groups:

        group = group.sort_values(TIME_COLUMN)

        series = (
            group["OBS_VALUE_SCALED"]
            .values
            .astype(np.float32)
        )

        if len(series) <= SEQUENCE_LENGTH:
            continue

        for i in range(len(series) - SEQUENCE_LENGTH):

            all_x.append(
                series[
                    i:i + SEQUENCE_LENGTH
                ]
            )

            all_y.append(
                series[
                    i + SEQUENCE_LENGTH
                ]
            )

            keys.append(key)

    x = (
        np.array(all_x)
        .reshape(-1, SEQUENCE_LENGTH, 1)
        .astype(np.float32)
    )

    y = (
        np.array(all_y)
        .reshape(-1, 1)
        .astype(np.float32)
    )

    return x, y, keys


def build_catboost_features(df):
    """
    Generate lag features.
    """

    feature_df = (
        df
        .copy()
        .sort_values(
            GROUP_COLUMNS + [TIME_COLUMN]
        )
        .reset_index(drop=True)
    )

    for lag in LAG_FEATURES:

        feature_df[f"LAG_{lag}"] = (
            feature_df
            .groupby(GROUP_COLUMNS)[TARGET_COLUMN]
            .shift(lag)
        )

    feature_df[
        ROLLING_MEAN_COLUMN
    ] = (
        feature_df
        .groupby(GROUP_COLUMNS)[TARGET_COLUMN]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                3,
                min_periods=1
            )
            .mean()
        )
    )

    feature_df[
        ROLLING_STD_COLUMN
    ] = (
        feature_df
        .groupby(GROUP_COLUMNS)[TARGET_COLUMN]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                3,
                min_periods=1
            )
            .std()
            .fillna(0)
        )
    )

    feature_df[
        YOY_COLUMN
    ] = (
        feature_df
        .groupby(GROUP_COLUMNS)[TARGET_COLUMN]
        .diff(1)
    )

    feature_df.dropna(inplace=True)

    feature_df.reset_index(
        drop=True,
        inplace=True,
    )

    return feature_df