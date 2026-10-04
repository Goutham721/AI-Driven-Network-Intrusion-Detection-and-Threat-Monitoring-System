"""
preprocessing.py
Shared preprocessing pipeline used identically by train_model.py and
predict.py. This module is the single source of truth for how raw
traffic records become model-ready feature vectors -- this is what
prevents train/predict mismatch, the most common bug in student IDS
projects.
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    ENCODERS_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    SCALER_PATH,
    TARGET_COLUMN,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and remove duplicate records."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d duplicate rows.", before - len(df))

    # Numeric columns: fill missing with median (robust to outliers)
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Filled missing values in '%s' with median (%.2f).", col, median_val)

    # Categorical columns: fill missing with mode
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("Filled missing values in '%s' with mode ('%s').", col, mode_val)

    return df


def fit_preprocessing_objects(df: pd.DataFrame):
    """
    Fit a StandardScaler on numeric features and LabelEncoders on
    categorical features + target, using TRAINING data only (avoids
    data leakage from the test set).
    """
    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_FEATURES])

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le

    label_encoder = LabelEncoder()
    label_encoder.fit(df[TARGET_COLUMN])
    encoders[TARGET_COLUMN] = label_encoder

    return scaler, encoders


def transform(df: pd.DataFrame, scaler: StandardScaler, encoders: dict,
              include_target: bool = True) -> pd.DataFrame:
    """
    Apply already-fitted scaler/encoders to a dataframe. Used identically
    at training time (on train/test splits) and prediction time (on a
    single live record), so behaviour never diverges.
    """
    df = df.copy()

    # Encode categorical features; unseen categories fall back to the
    # first known class index rather than crashing the pipeline.
    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        known_classes = set(le.classes_)
        df[col] = df[col].apply(lambda v: v if v in known_classes else le.classes_[0])
        df[col] = le.transform(df[col])

    # Scale numeric features
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])

    if include_target and TARGET_COLUMN in df.columns:
        le_target = encoders[TARGET_COLUMN]
        df[TARGET_COLUMN] = le_target.transform(df[TARGET_COLUMN])

    return df


def save_preprocessing_objects(scaler, encoders):
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    logger.info("Saved scaler to %s", SCALER_PATH)
    logger.info("Saved encoders to %s", ENCODERS_PATH)


def load_preprocessing_objects():
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return scaler, encoders


def prepare_training_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Full preprocessing entry point for train_model.py:
    clean -> fit on train -> transform train & test -> return X/y splits
    plus the fitted objects (which the caller should save).
    """
    train_df = clean_dataframe(train_df)
    test_df = clean_dataframe(test_df)

    scaler, encoders = fit_preprocessing_objects(train_df)

    train_processed = transform(train_df, scaler, encoders, include_target=True)
    test_processed = transform(test_df, scaler, encoders, include_target=True)

    X_train = train_processed[FEATURE_COLUMNS]
    y_train = train_processed[TARGET_COLUMN]
    X_test = test_processed[FEATURE_COLUMNS]
    y_test = test_processed[TARGET_COLUMN]

    return X_train, y_train, X_test, y_test, scaler, encoders
