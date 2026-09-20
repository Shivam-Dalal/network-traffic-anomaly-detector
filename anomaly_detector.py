"""
anomaly_detector.py

Unsupervised network traffic anomaly detector using Isolation Forest.

Approach:
- Treat each network flow as a point in feature space (packet stats, timing,
  port-touch behavior).
- Isolation Forest learns what "normal" looks like without needing labels
  at training time (mirrors a real SOC setting, where most traffic is
  unlabeled and attacks are rare/unknown in advance).
- Ground-truth labels (available here only because this is simulated data)
  are used purely for offline evaluation of detector quality.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from traffic_simulator import generate_dataset

FEATURES = [
    "protocol_encoded",
    "dst_port",
    "duration",
    "packet_count",
    "avg_packet_size",
    "bytes_total",
    "unique_dst_ports_from_src",
    "inter_arrival_ms",
]


def load_data():
    df = generate_dataset()
    df["is_attack"] = (df["label"] != "normal").astype(int)
    return df


def train_and_evaluate(contamination=0.08):
    df = load_data()
    X = df[FEATURES]
    y = df["is_attack"]

    X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
        X, y, df["label"], test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)

    # IsolationForest returns 1 = normal, -1 = anomaly; flip to match our 0/1 labels
    raw_preds = model.predict(X_test_scaled)
    preds = (raw_preds == -1).astype(int)

    scores = -model.score_samples(X_test_scaled)  # higher score = more anomalous

    print("=== Classification report (0=normal, 1=attack) ===")
    print(classification_report(y_test, preds, digits=3))

    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, preds))

    auc = roc_auc_score(y_test, scores)
    print(f"ROC-AUC: {auc:.3f}")

    # Per-attack-type detection rate (recall by attack category)
    results = X_test.copy()
    results["true_label"] = labels_test.values
    results["predicted_anomaly"] = preds
    print("\n=== Detection rate by traffic type ===")
    print(results.groupby("true_label")["predicted_anomaly"].mean().round(3))

    return model, scaler, auc


if __name__ == "__main__":
    train_and_evaluate()
