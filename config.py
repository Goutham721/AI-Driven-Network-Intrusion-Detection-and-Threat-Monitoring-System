"""
config.py
Central configuration for the AI-Driven Network Intrusion Detection System.
Keeping all paths, constants, and tunable parameters in one place avoids
scattering "magic numbers" throughout the codebase.
"""

import os

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

TRAINING_DATA_PATH = os.path.join(DATA_DIR, "network_training_data.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "network_test_data.csv")
SIMULATED_DATA_PATH = os.path.join(DATA_DIR, "simulated_traffic.csv")

STAGE1_MODEL_PATH = os.path.join(MODELS_DIR, "intrusion_detection_model.pkl")
STAGE2_MODEL_PATH = os.path.join(MODELS_DIR, "attack_classification_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

INTRUSION_LOG_PATH = os.path.join(OUTPUTS_DIR, "intrusion_logs.csv")
PDF_REPORT_PATH = os.path.join(OUTPUTS_DIR, "intrusion_report.pdf")

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
NUMBER_OF_RECORDS = 10000          # total synthetic records to generate
RANDOM_SEED = 42                   # reproducibility across the whole pipeline

# Class balance for the generated dataset (must sum to 1.0)
CLASS_DISTRIBUTION = {
    "Normal": 0.55,
    "DoS": 0.10,
    "DDoS": 0.10,
    "Port Scan": 0.10,
    "Brute Force": 0.10,
    "Suspicious Traffic": 0.05,
}

ATTACK_LABELS = [c for c in CLASS_DISTRIBUTION if c != "Normal"]
ALL_LABELS = list(CLASS_DISTRIBUTION.keys())

# ---------------------------------------------------------------------------
# Live traffic simulator
# ---------------------------------------------------------------------------
NORMAL_PROBABILITY = 0.70
ATTACK_PROBABILITY = 0.30
DEFAULT_SIMULATION_SPEED_SEC = 0.5   # delay between simulated records

# ---------------------------------------------------------------------------
# Threat severity thresholds (based on model confidence)
# ---------------------------------------------------------------------------
SEVERITY_THRESHOLDS = {
    "LOW": 0.50,
    "MEDIUM": 0.70,
    "HIGH": 0.85,
    "CRITICAL": 0.95,
}

# ---------------------------------------------------------------------------
# Feature columns used by the ML pipeline
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "source_port",
    "destination_port",
    "packet_count",
    "byte_count",
    "packet_size",
    "duration",
    "packets_per_second",
    "bytes_per_second",
    "connection_count",
    "failed_connections",
    "request_rate",
    "response_rate",
    "payload_size",
    "flow_rate",
]

CATEGORICAL_FEATURES = ["protocol", "tcp_flags"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "label"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
