"""
data_generator.py
Generates a labeled synthetic network traffic dataset for training the
intrusion detection models. Each attack type is given a distinct
statistical signature so that the ML models have real patterns to learn
from -- this is NOT random noise with random labels attached.

Run directly:
    python src/data_generator.py
"""

import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    CLASS_DISTRIBUTION,
    NUMBER_OF_RECORDS,
    RANDOM_SEED,
    TRAINING_DATA_PATH,
    TEST_DATA_PATH,
    DATA_DIR,
)
from src.utils import get_logger, random_ip, random_internal_ip, safe_makedirs  # noqa: E402

logger = get_logger(__name__)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROTOCOLS = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS"]
TCP_FLAGS = ["SYN", "ACK", "SYN-ACK", "FIN", "RST", "PSH-ACK"]


def _base_record():
    """Common fields present in every record regardless of traffic type."""
    return {
        "source_ip": random_ip(),
        "destination_ip": random_internal_ip(),
        "source_port": random.randint(1024, 65535),
        "destination_port": random.choice([80, 443, 22, 21, 53, 3389, 8080, 3306]),
        "protocol": random.choice(PROTOCOLS),
        "tcp_flags": random.choice(TCP_FLAGS),
    }


def generate_normal_record():
    """
    Normal traffic: moderate packet counts, reasonable duration,
    low failed-connection counts, balanced request/response rates.
    Represents everyday HTTP/HTTPS/DNS-like browsing and connections.
    """
    rec = _base_record()
    rec.update({
        "packet_count": np.random.randint(10, 300),
        "byte_count": np.random.randint(500, 50000),
        "packet_size": np.random.randint(64, 1500),
        "duration": round(np.random.uniform(0.5, 30.0), 2),
        "packets_per_second": round(np.random.uniform(1, 20), 2),
        "bytes_per_second": round(np.random.uniform(100, 5000), 2),
        "connection_count": np.random.randint(1, 5),
        "failed_connections": np.random.randint(0, 2),
        "request_rate": round(np.random.uniform(1, 10), 2),
        "response_rate": round(np.random.uniform(1, 10), 2),
        "payload_size": np.random.randint(100, 1400),
        "flow_rate": round(np.random.uniform(1, 15), 2),
        "label": "Normal",
    })
    return rec


def generate_dos_record():
    """
    DoS: single-source flood -- very high packet rate, high byte rate,
    long-ish duration, few distinct connections repeated aggressively.
    """
    rec = _base_record()
    rec.update({
        "packet_count": np.random.randint(5000, 50000),
        "byte_count": np.random.randint(500000, 5000000),
        "packet_size": np.random.randint(40, 200),
        "duration": round(np.random.uniform(10, 120), 2),
        "packets_per_second": round(np.random.uniform(500, 5000), 2),
        "bytes_per_second": round(np.random.uniform(50000, 500000), 2),
        "connection_count": np.random.randint(1, 3),
        "failed_connections": np.random.randint(0, 5),
        "request_rate": round(np.random.uniform(500, 3000), 2),
        "response_rate": round(np.random.uniform(0, 50), 2),
        "payload_size": np.random.randint(0, 100),
        "flow_rate": round(np.random.uniform(500, 4000), 2),
        "label": "DoS",
    })
    return rec


def generate_ddos_record():
    """
    DDoS: many simulated distinct source IPs, high request rate,
    high connection frequency, abnormal packet distribution.
    """
    rec = _base_record()
    rec.update({
        "packet_count": np.random.randint(2000, 20000),
        "byte_count": np.random.randint(200000, 2000000),
        "packet_size": np.random.randint(40, 300),
        "duration": round(np.random.uniform(1, 15), 2),
        "packets_per_second": round(np.random.uniform(300, 3000), 2),
        "bytes_per_second": round(np.random.uniform(30000, 300000), 2),
        "connection_count": np.random.randint(50, 500),
        "failed_connections": np.random.randint(10, 100),
        "request_rate": round(np.random.uniform(300, 2500), 2),
        "response_rate": round(np.random.uniform(0, 30), 2),
        "payload_size": np.random.randint(0, 150),
        "flow_rate": round(np.random.uniform(400, 3500), 2),
        "label": "DDoS",
    })
    return rec


def generate_port_scan_record():
    """
    Port Scan: many destination ports probed, short connection duration,
    many failed connections, very low payload size.
    """
    rec = _base_record()
    rec.update({
        "destination_port": random.randint(1, 65535),  # scans many ports
        "packet_count": np.random.randint(1, 10),
        "byte_count": np.random.randint(40, 500),
        "packet_size": np.random.randint(40, 80),
        "duration": round(np.random.uniform(0.01, 0.5), 3),
        "packets_per_second": round(np.random.uniform(50, 400), 2),
        "bytes_per_second": round(np.random.uniform(500, 5000), 2),
        "connection_count": np.random.randint(20, 200),
        "failed_connections": np.random.randint(15, 180),
        "request_rate": round(np.random.uniform(50, 300), 2),
        "response_rate": round(np.random.uniform(0, 5), 2),
        "payload_size": np.random.randint(0, 20),
        "flow_rate": round(np.random.uniform(50, 350), 2),
        "label": "Port Scan",
    })
    return rec


def generate_brute_force_record():
    """
    Brute Force: repeated connection attempts against one service/port,
    high failed-connection ratio, elevated request frequency.
    """
    rec = _base_record()
    rec.update({
        "destination_port": random.choice([22, 21, 3389, 3306, 443]),
        "packet_count": np.random.randint(50, 800),
        "byte_count": np.random.randint(2000, 40000),
        "packet_size": np.random.randint(60, 400),
        "duration": round(np.random.uniform(1, 20), 2),
        "packets_per_second": round(np.random.uniform(20, 150), 2),
        "bytes_per_second": round(np.random.uniform(2000, 20000), 2),
        "connection_count": np.random.randint(30, 300),
        "failed_connections": np.random.randint(25, 280),
        "request_rate": round(np.random.uniform(20, 150), 2),
        "response_rate": round(np.random.uniform(5, 30), 2),
        "payload_size": np.random.randint(20, 300),
        "flow_rate": round(np.random.uniform(20, 140), 2),
        "label": "Brute Force",
    })
    return rec


def generate_suspicious_record():
    """
    Suspicious Traffic: doesn't cleanly match a known attack signature
    but deviates moderately from normal baselines -- useful for
    demonstrating LOW/MEDIUM severity, lower-confidence predictions.
    """
    rec = _base_record()
    rec.update({
        "packet_count": np.random.randint(300, 1500),
        "byte_count": np.random.randint(50000, 300000),
        "packet_size": np.random.randint(100, 900),
        "duration": round(np.random.uniform(5, 60), 2),
        "packets_per_second": round(np.random.uniform(20, 100), 2),
        "bytes_per_second": round(np.random.uniform(5000, 30000), 2),
        "connection_count": np.random.randint(5, 25),
        "failed_connections": np.random.randint(3, 15),
        "request_rate": round(np.random.uniform(10, 60), 2),
        "response_rate": round(np.random.uniform(5, 40), 2),
        "payload_size": np.random.randint(100, 1000),
        "flow_rate": round(np.random.uniform(15, 80), 2),
        "label": "Suspicious Traffic",
    })
    return rec


GENERATORS = {
    "Normal": generate_normal_record,
    "DoS": generate_dos_record,
    "DDoS": generate_ddos_record,
    "Port Scan": generate_port_scan_record,
    "Brute Force": generate_brute_force_record,
    "Suspicious Traffic": generate_suspicious_record,
}


def generate_dataset(n_records: int = NUMBER_OF_RECORDS) -> pd.DataFrame:
    """Generate a full labeled dataset according to CLASS_DISTRIBUTION."""
    records = []
    for label, proportion in CLASS_DISTRIBUTION.items():
        count = int(n_records * proportion)
        generator_fn = GENERATORS[label]
        for _ in range(count):
            records.append(generator_fn())

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle
    logger.info("Generated %d synthetic traffic records.", len(df))
    logger.info("Class distribution:\n%s", df["label"].value_counts())
    return df


def main():
    safe_makedirs(DATA_DIR)
    df = generate_dataset(NUMBER_OF_RECORDS)

    # 80/20 train/test split, stratified by label to keep class balance
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
    )

    train_df.to_csv(TRAINING_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    logger.info("Saved training data (%d rows) to %s", len(train_df), TRAINING_DATA_PATH)
    logger.info("Saved test data (%d rows) to %s", len(test_df), TEST_DATA_PATH)


if __name__ == "__main__":
    main()
