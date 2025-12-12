
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_logs(path: str = "logs/metrics.jsonl") -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Log dosyası bulunamadı: {p}")

    records = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize_detect_logs(records: List[Dict]) -> Dict:
    latencies = [
        r["latency_ms"] for r in records if r.get("event") == "detect"
    ]
    if not latencies:
        return {}

    import numpy as np

    arr = np.array(latencies)
    return {
        "count": int(arr.size),
            "mean_latency_ms": float(arr.mean()),
            "p50_latency_ms": float(np.percentile(arr, 50)),
            "p95_latency_ms": float(np.percentile(arr, 95)),
    }


if __name__ == "__main__":
    logs = load_logs()
    summary = summarize_detect_logs(logs)
    print("Detection latency summary:")
    print(json.dumps(summary, indent=2))
