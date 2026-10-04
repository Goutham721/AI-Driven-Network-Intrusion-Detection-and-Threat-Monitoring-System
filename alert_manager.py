"""
alert_manager.py
Creates and stores intrusion alert/log entries, and provides simple
filtering helpers used by the dashboard's Intrusion Logs page.

Security actions here are simulated only -- this module never touches
real network connections. See SECURITY_ACTIONS below.
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INTRUSION_LOG_PATH, OUTPUTS_DIR  # noqa: E402
from src.utils import current_timestamp, safe_makedirs  # noqa: E402

LOG_COLUMNS = [
    "event_id",
    "timestamp",
    "source_ip",
    "destination_ip",
    "source_port",
    "destination_port",
    "protocol",
    "attack_type",
    "confidence",
    "severity",
    "action",
]

SEVERITY_ACTION_MAP = {
    "LOW": "MONITOR",
    "MEDIUM": "ALERT",
    "HIGH": "FLAG SOURCE",
    "CRITICAL": "SIMULATED BLOCK",
}


def determine_action(severity: str) -> str:
    """Map severity to a simulated security response (never a real action)."""
    return SEVERITY_ACTION_MAP.get(severity, "MONITOR")


def create_alert_entry(record: dict, prediction: dict, event_id: int) -> dict:
    """Build a single log/alert row from a traffic record + its prediction."""
    return {
        "event_id": event_id,
        "timestamp": current_timestamp(),
        "source_ip": record.get("source_ip"),
        "destination_ip": record.get("destination_ip"),
        "source_port": record.get("source_port"),
        "destination_port": record.get("destination_port"),
        "protocol": record.get("protocol"),
        "attack_type": prediction.get("attack_type") or "N/A",
        "confidence": prediction.get("confidence"),
        "severity": prediction.get("severity"),
        "action": determine_action(prediction.get("severity", "LOW")),
    }


def append_to_log(log_df: pd.DataFrame, entry: dict) -> pd.DataFrame:
    """Append a new entry to the in-memory log DataFrame."""
    new_row = pd.DataFrame([entry])
    return pd.concat([log_df, new_row], ignore_index=True)


def save_log(log_df: pd.DataFrame) -> None:
    safe_makedirs(OUTPUTS_DIR)
    log_df.to_csv(INTRUSION_LOG_PATH, index=False)


def load_log() -> pd.DataFrame:
    if os.path.exists(INTRUSION_LOG_PATH):
        return pd.read_csv(INTRUSION_LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def filter_log(log_df: pd.DataFrame, attack_type=None, severity=None,
                source_ip=None) -> pd.DataFrame:
    """Filter the intrusion log by attack type, severity, and/or source IP."""
    filtered = log_df.copy()
    if attack_type and attack_type != "All":
        filtered = filtered[filtered["attack_type"] == attack_type]
    if severity and severity != "All":
        filtered = filtered[filtered["severity"] == severity]
    if source_ip:
        filtered = filtered[filtered["source_ip"].str.contains(source_ip, na=False)]
    return filtered
