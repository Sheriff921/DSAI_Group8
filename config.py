"""
config.py

Global configuration file for the AI-Based Energy Consumption Forecasting
dashboard.

All paths, constants, feature names and model settings are defined here so
other modules never need hardcoded values.
"""

from pathlib import Path

# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# =============================================================================
# DIRECTORIES
# =============================================================================

SOURCE_DIR = PROJECT_ROOT / "Source"
DATASET_DIR = SOURCE_DIR / "Dataset"
PAPER_DIR = SOURCE_DIR / "PaperReference"

OUTPUT_DIR = PROJECT_ROOT / "Output"
MODEL_DIR = OUTPUT_DIR / "models"
PREPROCESS_DIR = OUTPUT_DIR / "Model"

EDA_DIR = PROJECT_ROOT / "EDA"
POST_EDA_DIR = PROJECT_ROOT / "Postprocessing EDA"

LOG_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# DATASETS
# =============================================================================

RAW_DATASET = DATASET_DIR / "WB_SE4ALL.csv"
PROCESSED_DATASET = DATASET_DIR / "processed_data_v2.csv"

# =============================================================================
# TRAINED MODELS
# =============================================================================

CATBOOST_MODEL = MODEL_DIR / "catboost_model.cbm"
GRU_MODEL = MODEL_DIR / "gru_model.pt"
PROPHET_MODELS = MODEL_DIR / "prophet_models.pkl"
PROPHET_SINGLE = MODEL_DIR / "prophet_single_series.pkl"

# =============================================================================
# PREPROCESSING FILES
# =============================================================================

ENCODER_FILE = PREPROCESS_DIR / "encoders.pkl"
SCALE_PARAM_FILE = PREPROCESS_DIR / "scale_params.pkl"

# =============================================================================
# IMAGES
# =============================================================================

EDA_IMAGES = {
    "Target Distribution": EDA_DIR / "target_distribution.png",
    "Correlation Heatmap": EDA_DIR / "correlation_heatmap.png",
    "Missing Values": EDA_DIR / "missing_values.png",
    "OBS_VALUE Distribution": EDA_DIR / "OBS_VALUE_distribution.png",
    "OBS_VALUE Boxplot": EDA_DIR / "OBS_VALUE_boxplot.png",
    "TIME_PERIOD Distribution": EDA_DIR / "TIME_PERIOD_distribution.png",
    "TIME_PERIOD Boxplot": EDA_DIR / "TIME_PERIOD_boxplot.png",
}

POST_EDA_IMAGES = {
    "Feature Distribution": POST_EDA_DIR / "feature_distributions.png",
    "Feature Boxplots": POST_EDA_DIR / "feature_boxplots.png",
    "Missing Values Check": POST_EDA_DIR / "missing_values_check.png",
}

MODEL_COMPARISON_IMAGE = (
    OUTPUT_DIR
    / "model_evaluation_comparison"
    / "model_comparison_final.png"
)

# =============================================================================
# DATA SPLITTING
# =============================================================================

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

# =============================================================================
# GRU SETTINGS
# =============================================================================

SEQUENCE_LENGTH = 3

# =============================================================================
# CATBOOST SETTINGS
# =============================================================================

LAG_FEATURES = [1, 2, 3]

ROLLING_MEAN_COLUMN = "ROLLING_MEAN_3"
ROLLING_STD_COLUMN = "ROLLING_STD_3"
YOY_COLUMN = "YOY_DIFF"

# =============================================================================
# FEATURE COLUMNS
# =============================================================================

GROUP_COLUMNS = [
    "COUNTRY_ID",
    "INDICATOR_ID",
    "COMP_ID",
]

TARGET_COLUMN = "OBS_VALUE"

SCALED_TARGET = "OBS_VALUE_SCALED"

TIME_COLUMN = "TIME_PERIOD"

DATE_COLUMN = "DATE_STAMPS"

# =============================================================================
# APP SETTINGS
# =============================================================================

APP_TITLE = "AI-based Energy Consumption Forecasting"

APP_ICON = "⚡"

LAYOUT = "wide"

# =============================================================================
# FILE VALIDATION
# =============================================================================

REQUIRED_FILES = [
    RAW_DATASET,
    PROCESSED_DATASET,
    CATBOOST_MODEL,
    GRU_MODEL,
    PROPHET_MODELS,
    ENCODER_FILE,
    SCALE_PARAM_FILE,
]


def validate_project_structure():
    """
    Check whether required project files exist.
    """

    missing = []

    for file in REQUIRED_FILES:
        if not file.exists():
            missing.append(file)

    return missing
