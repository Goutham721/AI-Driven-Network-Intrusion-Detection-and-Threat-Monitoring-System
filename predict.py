"""
predict.py
Runs a single traffic record through the trained two-stage pipeline and
returns a structured prediction. Uses the exact same preprocessing
objects (scaler/encoders) that were fit during training -- never refits
anything here.
"""

import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    FEATURE_COLUMNS,
    STAGE1_MODEL_PATH,
    STAGE2_MODEL_PATH,
)
from src.preprocessing import load_preprocessing_objects, transform  # noqa: E402
from src.threat_analyzer import determine_severity  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

_stage1_model = None
_stage2_model = None
_scaler = None
_encoders = None


def _load_artifacts():
    """Lazily load models + preprocessing objects (cached across calls)."""
    global _stage1_model, _stage2_model, _scaler, _encoders

    if _stage1_model is None:
        if not os.path.exists(STAGE1_MODEL_PATH):
            raise FileNotFoundError(
                f"Stage 1 model not found at {STAGE1_MODEL_PATH}. "
                "Run train_model.py first."
            )
        _stage1_model = joblib.load(STAGE1_MODEL_PATH)

    if _stage2_model is None:
        if not os.path.exists(STAGE2_MODEL_PATH):
            raise FileNotFoundError(
                f"Stage 2 model not found at {STAGE2_MODEL_PATH}. "
                "Run train_model.py first."
            )
        _stage2_model = joblib.load(STAGE2_MODEL_PATH)

    if _scaler is None or _encoders is None:
        _scaler, _encoders = load_preprocessing_objects()

    return _stage1_model, _stage2_model, _scaler, _encoders


def predict_record(record: dict) -> dict:
    """
    Predict on a single traffic record (dict of raw feature values).

    Returns a structured dict, e.g.:
        {
            "status": "ATTACK",
            "attack_type": "Port Scan",
            "confidence": 0.94,
            "severity": "HIGH",
            "contributing_indicators": [...]
        }
    """
    stage1_model, stage2_model, scaler, encoders = _load_artifacts()

    df = pd.DataFrame([record])
    processed = transform(df, scaler, encoders, include_target=False)
    X = processed[FEATURE_COLUMNS]

    # Stage 1: Normal vs Attack
    stage1_pred = stage1_model.predict(X)[0]
    stage1_proba = stage1_model.predict_proba(X)[0]
    stage1_confidence = float(max(stage1_proba))

    if stage1_pred == 0:
        return {
            "status": "NORMAL",
            "attack_type": None,
            "confidence": round(stage1_confidence, 4),
            "severity": "NONE",
            "contributing_indicators": [],
        }

    # Stage 2: Attack type classification
    stage2_pred = stage2_model.predict(X)[0]
    stage2_proba = stage2_model.predict_proba(X)[0]
    stage2_confidence = float(max(stage2_proba))

    severity = determine_severity(stage2_confidence, record)
    indicators = _contributing_indicators(record, stage2_pred)

    return {
        "status": "ATTACK",
        "attack_type": stage2_pred,
        "confidence": round(stage2_confidence, 4),
        "severity": severity,
        "contributing_indicators": indicators,
    }


def _contributing_indicators(record: dict, attack_type: str) -> list:
    """
    Rule-based explainability: surfaces which raw traffic characteristics
    most plausibly drove this classification. This is a simple, honest
    heuristic layer -- NOT a SHAP explanation. It is labeled as such in
    the dashboard.
    """
    indicators = []

    if attack_type == "Port Scan":
        if record.get("failed_connections", 0) > 15:
            indicators.append("High number of failed connection attempts")
        if record.get("connection_count", 0) > 20:
            indicators.append("Unusually high number of distinct connections")
        if record.get("payload_size", 0) < 20:
            indicators.append("Very low payload size")
        if record.get("duration", 0) < 0.5:
            indicators.append("Extremely short connection duration")

    elif attack_type in ("DoS", "DDoS"):
        if record.get("packets_per_second", 0) > 300:
            indicators.append("Very high packets-per-second rate")
        if record.get("bytes_per_second", 0) > 30000:
            indicators.append("Very high bytes-per-second rate")
        if record.get("connection_count", 0) > 50:
            indicators.append("High number of simultaneous connections")
        if record.get("request_rate", 0) > 300:
            indicators.append("Abnormally high request rate")

    elif attack_type == "Brute Force":
        if record.get("failed_connections", 0) > 25:
            indicators.append("High ratio of failed connection attempts")
        if record.get("connection_count", 0) > 30:
            indicators.append("Repeated connection attempts to same service")
        if record.get("destination_port") in (22, 21, 3389, 3306):
            indicators.append("Targeting a common authentication service port")

    elif attack_type == "Suspicious Traffic":
        indicators.append("Traffic pattern deviates moderately from normal baseline")

    if not indicators:
        indicators.append("Traffic pattern statistically deviates from learned normal behavior")

    return indicators
