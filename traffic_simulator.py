"""
traffic_simulator.py

Generates simulated network flow records for anomaly detection experiments.

Why simulated data: capturing real enterprise traffic requires raw-socket
access and a live network, which isn't available in a dev/sandbox setting.
Using simulated flows with realistic feature distributions (modeled after
known patterns: normal HTTP/HTTPS browsing vs. port-scan / DoS / exfiltration
behavior) is a standard way to build and validate a detector's logic before
pointing it at a live capture (see live_capture.py for the real-traffic path).

Each record is a "flow": an aggregation of packets between two endpoints,
similar in spirit to NetFlow/CICFlowMeter-style features.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

PROTOCOLS = {"TCP": 0, "UDP": 1, "ICMP": 2}
COMMON_PORTS = [80, 443, 22, 53, 8080]


def _normal_flows(n):
    """Typical benign traffic: web browsing, DNS, SSH sessions."""
    protocol = RNG.choice(["TCP", "UDP"], size=n, p=[0.85, 0.15])
    dst_port = RNG.choice(COMMON_PORTS, size=n, p=[0.4, 0.35, 0.1, 0.1, 0.05])
    duration = RNG.gamma(shape=2.0, scale=1.5, size=n)  # seconds
    packet_count = RNG.poisson(lam=25, size=n) + 1
    avg_packet_size = RNG.normal(loc=650, scale=150, size=n).clip(40, 1500)
    bytes_total = packet_count * avg_packet_size
    unique_dst_ports_from_src = RNG.integers(1, 3, size=n)  # rarely touches many ports
    inter_arrival_ms = RNG.normal(loc=120, scale=40, size=n).clip(1, None)

    return pd.DataFrame({
        "protocol": protocol,
        "dst_port": dst_port,
        "duration": duration,
        "packet_count": packet_count,
        "avg_packet_size": avg_packet_size,
        "bytes_total": bytes_total,
        "unique_dst_ports_from_src": unique_dst_ports_from_src,
        "inter_arrival_ms": inter_arrival_ms,
        "label": "normal",
    })


def _port_scan_flows(n):
    """Attacker touching many ports in short bursts, tiny packets."""
    protocol = np.full(n, "TCP")
    dst_port = RNG.integers(1, 65535, size=n)
    duration = RNG.uniform(0.01, 0.3, size=n)
    packet_count = RNG.integers(1, 3, size=n)
    avg_packet_size = RNG.normal(loc=60, scale=10, size=n).clip(40, 100)
    bytes_total = packet_count * avg_packet_size
    unique_dst_ports_from_src = RNG.integers(30, 500, size=n)
    inter_arrival_ms = RNG.uniform(0.1, 5, size=n)

    return pd.DataFrame({
        "protocol": protocol,
        "dst_port": dst_port,
        "duration": duration,
        "packet_count": packet_count,
        "avg_packet_size": avg_packet_size,
        "bytes_total": bytes_total,
        "unique_dst_ports_from_src": unique_dst_ports_from_src,
        "inter_arrival_ms": inter_arrival_ms,
        "label": "port_scan",
    })


def _dos_flows(n):
    """Volumetric flood: huge packet counts to one service, short duration."""
    protocol = RNG.choice(["TCP", "UDP"], size=n)
    dst_port = RNG.choice([80, 443, 53], size=n)
    duration = RNG.uniform(1, 10, size=n)
    packet_count = RNG.integers(5000, 50000, size=n)
    avg_packet_size = RNG.normal(loc=100, scale=20, size=n).clip(40, 200)
    bytes_total = packet_count * avg_packet_size
    unique_dst_ports_from_src = np.ones(n, dtype=int)
    inter_arrival_ms = RNG.uniform(0.01, 0.5, size=n)

    return pd.DataFrame({
        "protocol": protocol,
        "dst_port": dst_port,
        "duration": duration,
        "packet_count": packet_count,
        "avg_packet_size": avg_packet_size,
        "bytes_total": bytes_total,
        "unique_dst_ports_from_src": unique_dst_ports_from_src,
        "inter_arrival_ms": inter_arrival_ms,
        "label": "dos",
    })


def _exfiltration_flows(n):
    """Large outbound transfer to an unusual port, long steady duration."""
    protocol = np.full(n, "TCP")
    dst_port = RNG.integers(1024, 65535, size=n)
    duration = RNG.uniform(60, 600, size=n)
    packet_count = RNG.integers(2000, 20000, size=n)
    avg_packet_size = RNG.normal(loc=1400, scale=80, size=n).clip(500, 1500)
    bytes_total = packet_count * avg_packet_size
    unique_dst_ports_from_src = np.ones(n, dtype=int)
    inter_arrival_ms = RNG.normal(loc=15, scale=5, size=n).clip(1, None)

    return pd.DataFrame({
        "protocol": protocol,
        "dst_port": dst_port,
        "duration": duration,
        "packet_count": packet_count,
        "avg_packet_size": avg_packet_size,
        "bytes_total": bytes_total,
        "unique_dst_ports_from_src": unique_dst_ports_from_src,
        "inter_arrival_ms": inter_arrival_ms,
        "label": "exfiltration",
    })


def generate_dataset(n_normal=4000, n_attack_each=120):
    """
    Returns a DataFrame of flows. Attack traffic is ~8% of total,
    which is a deliberately realistic ratio for anomaly detection
    (attacks are rare events, not 50/50 with normal traffic).
    """
    frames = [
        _normal_flows(n_normal),
        _port_scan_flows(n_attack_each),
        _dos_flows(n_attack_each),
        _exfiltration_flows(n_attack_each),
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    df["protocol_encoded"] = df["protocol"].map(PROTOCOLS).fillna(1)
    return df


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv("simulated_traffic.csv", index=False)
    print(f"Generated {len(data)} flow records -> simulated_traffic.csv")
    print(data["label"].value_counts())
