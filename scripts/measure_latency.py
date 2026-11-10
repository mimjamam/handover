"""Measure response time of a running service on real test tickets.

Run:  uv run python scripts/measure_latency.py [base_url]
Writes reports/latency.json.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import requests

from handover.data import load

N = 300


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    texts = load().query("split == 'test'")["text"].sample(N, random_state=0).tolist()
    requests.post(f"{base}/route", json={"text": texts[0]}, timeout=120)  # warm-up

    total, model = [], []
    for text in texts:
        start = time.perf_counter()
        r = requests.post(f"{base}/route", json={"text": text[:5000]}, timeout=120)
        r.raise_for_status()
        total.append((time.perf_counter() - start) * 1000)
        model.append(r.json()["latency_ms"])

    def summary(values: list[float]) -> dict:
        return {f"p{q}": round(float(np.percentile(values, q)), 1) for q in (50, 95, 99)}

    result = {
        "requests": N,
        "service": base,
        "machine": f"{platform.processor() or platform.machine()}, {__import__('os').cpu_count()} logical CPUs",
        "end_to_end_ms": summary(total),
        "model_only_ms": summary(model),
        "sequential_requests_per_second": round(1000 / float(np.mean(total)), 1),
    }
    Path("reports/latency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
