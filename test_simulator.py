"""
test_simulator.py
Tests for traffic_simulator.py: ensures generated records contain the
expected fields and that probability weighting behaves sensibly.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_COLUMNS
from src.traffic_simulator import generate_live_record


def test_generated_record_has_required_fields():
    record = generate_live_record(attack_probability=0.5)
    for col in FEATURE_COLUMNS:
        assert col in record
    assert "true_label" in record
    assert "source_ip" in record
    assert "destination_ip" in record


def test_zero_attack_probability_yields_mostly_normal():
    labels = [generate_live_record(attack_probability=0.0)["true_label"] for _ in range(50)]
    assert labels.count("Normal") == 50


def test_full_attack_probability_yields_no_normal():
    labels = [generate_live_record(attack_probability=1.0)["true_label"] for _ in range(50)]
    assert "Normal" not in labels
