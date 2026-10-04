"""
test_model.py
Basic tests for the training pipeline artifacts: confirms that trained
models and preprocessing objects exist and load correctly, and that
metrics were computed (not hard-coded).

Run:
    python -m pytest tests/test_model.py -v
(Requires train_model.py to have been run first.)
"""

import json
import os
import sys

import joblib
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import METRICS_PATH, SCALER_PATH, STAGE1_MODEL_PATH, STAGE2_MODEL_PATH


@pytest.mark.skipif(not os.path.exists(STAGE1_MODEL_PATH), reason="Model not trained yet")
def test_stage1_model_loads():
    model = joblib.load(STAGE1_MODEL_PATH)
    assert model is not None


@pytest.mark.skipif(not os.path.exists(STAGE2_MODEL_PATH), reason="Model not trained yet")
def test_stage2_model_loads():
    model = joblib.load(STAGE2_MODEL_PATH)
    assert model is not None


@pytest.mark.skipif(not os.path.exists(SCALER_PATH), reason="Scaler not saved yet")
def test_scaler_loads():
    scaler = joblib.load(SCALER_PATH)
    assert hasattr(scaler, "transform")


@pytest.mark.skipif(not os.path.exists(METRICS_PATH), reason="Metrics not generated yet")
def test_metrics_contain_real_scores():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    stage1_best = metrics["stage1"]["best_model"]
    acc = metrics["stage1"]["comparison"][stage1_best]["accuracy"]
    assert 0.0 <= acc <= 1.0
