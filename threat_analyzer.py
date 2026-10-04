"""
threat_analyzer.py
Assigns an application-level threat severity (LOW/MEDIUM/HIGH/CRITICAL)
based primarily on model confidence, with a small boost for records
showing extreme traffic characteristics (e.g. very high packet rate).

NOTE: This severity score is an application-level risk indicator for
demonstration purposes -- it is NOT a standardized cybersecurity rating
such as CVSS.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEVERITY_THRESHOLDS  # noqa: E402


def determine_severity(confidence: float, record: dict) -> str:
    """
    Determine severity level from model confidence, escalated one level
    if the traffic shows an extreme volumetric signature (very high
    packet rate or failed-connection count), which often indicates a
    more aggressive/automated attack.
    """
    if confidence >= SEVERITY_THRESHOLDS["CRITICAL"]:
        severity = "CRITICAL"
    elif confidence >= SEVERITY_THRESHOLDS["HIGH"]:
        severity = "HIGH"
    elif confidence >= SEVERITY_THRESHOLDS["MEDIUM"]:
        severity = "MEDIUM"
    elif confidence >= SEVERITY_THRESHOLDS["LOW"]:
        severity = "LOW"
    else:
        severity = "LOW"

    # Escalate one level for extreme volumetric behavior
    extreme_volume = (
        record.get("packets_per_second", 0) > 1000
        or record.get("failed_connections", 0) > 100
    )
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if extreme_volume and severity != "CRITICAL":
        idx = order.index(severity)
        severity = order[min(idx + 1, len(order) - 1)]

    return severity


def severity_icon(severity: str) -> str:
    """Return a small colored-circle icon for the given severity level."""
    mapping = {
        "NONE": "🟢",
        "LOW": "🟡",
        "MEDIUM": "🟠",
        "HIGH": "🔴",
        "CRITICAL": "🔴",
    }
    return mapping.get(severity, "⚪")
