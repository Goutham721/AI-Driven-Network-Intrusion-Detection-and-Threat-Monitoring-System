"""
traffic_simulator.py
Generates a continuous stream of SIMULATED network traffic records for
the live dashboard demonstration. This is entirely synthetic --
no real network interfaces are touched and no real traffic is captured.

Clearly labeled throughout the dashboard as "Simulated Network Traffic".
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ATTACK_LABELS, ATTACK_PROBABILITY, NORMAL_PROBABILITY  # noqa: E402
from src.data_generator import GENERATORS  # noqa: E402


def generate_live_record(attack_probability: float = ATTACK_PROBABILITY) -> dict:
    """
    Generate a single simulated traffic record, weighted by the given
    attack probability. The returned record includes a 'true_label'
    (ground truth from the simulator, useful for demo/QA purposes) in
    addition to the feature fields the model will consume.
    """
    normal_probability = 1.0 - attack_probability

    if random.random() < normal_probability:
        label = "Normal"
    else:
        label = random.choice(ATTACK_LABELS)

    record = GENERATORS[label]()
    record["true_label"] = record.pop("label")
    return record
