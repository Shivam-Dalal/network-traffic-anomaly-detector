# network-traffic-anomaly-detector

An unsupervised machine learning system for flagging suspicious network flows
(port scans, DoS/flood traffic, data exfiltration patterns) without relying
on labeled attack signatures — using an Isolation Forest trained on flow-level
statistical features.

## Why this approach

Signature-based detection only catches known attack patterns. This project
instead learns what *normal* traffic looks like and flags statistical
outliers, which is the same core idea behind real anomaly-based intrusion
detection systems (and ties into the broader "AI-enabled security monitoring"
approach many security teams are adopting).

## What's real vs. simulated (read this before discussing the project)

- **Training/evaluation data is simulated**, generated with realistic feature
  distributions for benign traffic (web/DNS/SSH-like patterns) and three
  attack types (port scan, volumetric DoS, exfiltration). This was a
  deliberate choice: capturing real enterprise traffic requires raw-socket
  access and a live network that weren't available in the dev environment
  this was built in, and simulated data lets the detection logic be built
  and validated first.
- **`live_capture.py` is real, working Scapy code** for extracting the same
  features from actual packet captures, but it has not yet been run against
  live traffic (needs root/admin + a live NIC to test). It's the intended
  next step, not something to claim as already validated on real traffic.

## Results (on held-out simulated test data)

| Metric | Value |
|---|---|
| Overall accuracy | 99.4% |
| Attack recall (overall) | 92.6% |
| Attack precision | 100% (no false positives on this test set) |
| ROC-AUC | 1.00 |
| Detection rate — port scan | 87.8% |
| Detection rate — DoS | 90.0% |
| Detection rate — exfiltration | 100% |

(Full run output in `results_log.txt`.)

## Architecture

```
traffic_simulator.py   → generates labeled synthetic flow data
anomaly_detector.py    → feature scaling + Isolation Forest training/eval
live_capture.py        → Scapy-based real packet capture (not yet live-tested)
```

## Features used per flow

Protocol, destination port, duration, packet count, average packet size,
total bytes, number of unique destination ports touched by the source
(scan indicator), and mean inter-arrival time.

## How to run

```bash
pip install scikit-learn pandas numpy scapy
python3 anomaly_detector.py
```

## Honest next steps

- Validate against a real public dataset (e.g., CICIDS2017) instead of only
  simulated data
- Test `live_capture.py` against real traffic on a home network
- Add a supervised comparison model (e.g., Random Forest) once labeled real
  data is available, to benchmark against the unsupervised approach
