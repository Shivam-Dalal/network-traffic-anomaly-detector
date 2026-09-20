"""
live_capture.py

Real packet capture and flow feature extraction using Scapy, for pointing
the trained anomaly detector at actual network traffic instead of the
simulated dataset.

NOTE / HONESTY FLAG FOR README AND ANY INTERVIEW DISCUSSION:
This module requires raw-socket access (root on Linux/macOS, or Npcap on
Windows) and a live network interface. It has NOT been run against real
traffic in the environment this project was built in (a sandboxed container
with no raw-socket/live-NIC access) — it's included as the intended
production path and is straightforward to test on a laptop with:

    sudo python3 live_capture.py --iface eth0 --duration 60

Be upfront about this distinction if asked: the model was trained and
evaluated on simulated flow data (see traffic_simulator.py); this module
is the not-yet-live-tested extension for real packet capture.
"""

import argparse
import time
from collections import defaultdict

try:
    from scapy.all import sniff, IP, TCP, UDP
except ImportError as e:
    raise SystemExit(
        "Scapy is required for live capture: pip install scapy"
    ) from e


class FlowAggregator:
    """Buckets raw packets into flow-level features matching FEATURES
    in anomaly_detector.py, keyed by (src, dst, dst_port, protocol)."""

    def __init__(self):
        self.flows = defaultdict(lambda: {
            "packet_count": 0,
            "byte_total": 0,
            "first_seen": None,
            "last_seen": None,
            "dst_ports_seen": set(),
        })

    def add_packet(self, pkt):
        if IP not in pkt:
            return
        proto = "TCP" if TCP in pkt else ("UDP" if UDP in pkt else "OTHER")
        if proto == "OTHER":
            return

        src = pkt[IP].src
        dst_port = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport
        key = (src, proto)

        flow = self.flows[key]
        flow["packet_count"] += 1
        flow["byte_total"] += len(pkt)
        flow["dst_ports_seen"].add(dst_port)
        now = time.time()
        flow["first_seen"] = flow["first_seen"] or now
        flow["last_seen"] = now
        flow["protocol"] = proto
        flow["last_dst_port"] = dst_port

    def to_feature_rows(self):
        rows = []
        for (src, proto), f in self.flows.items():
            duration = max((f["last_seen"] - f["first_seen"]), 0.001)
            rows.append({
                "src": src,
                "protocol": f["protocol"],
                "dst_port": f["last_dst_port"],
                "duration": duration,
                "packet_count": f["packet_count"],
                "avg_packet_size": f["byte_total"] / f["packet_count"],
                "bytes_total": f["byte_total"],
                "unique_dst_ports_from_src": len(f["dst_ports_seen"]),
                "inter_arrival_ms": (duration * 1000) / max(f["packet_count"], 1),
            })
        return rows


def capture(iface, duration):
    agg = FlowAggregator()
    print(f"Capturing on {iface} for {duration}s (requires root/admin)...")
    sniff(iface=iface, timeout=duration, prn=agg.add_packet, store=False)
    return agg.to_feature_rows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()

    rows = capture(args.iface, args.duration)
    print(f"Captured {len(rows)} flows.")
    for r in rows[:10]:
        print(r)
