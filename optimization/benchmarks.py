
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil


def run_onnx_benchmark(
    model_path: str = "models/model.onnx",
    runs: int = 50,
    warmup: int = 10,
    img_size: int = 640,
    output_path: str = "optimization/benchmark_results.json",
) -> None:
    """
    ONNX modeli için basit latency + FPS benchmark'ı.
    Dummy input ile çalışır.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX modeli bulunamadı: {model_path}")

    print(f"[benchmarks] ONNX Runtime session oluşturuluyor: {model_path}")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    # Dummy input: 1x3xHxW
    dummy_input = np.random.rand(1, 3, img_size, img_size).astype(np.float32)

    # Warmup
    print(f"[benchmarks] Warmup ({warmup} iter)...")
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    # Benchmark
    print(f"[benchmarks] Benchmark ({runs} iter)...")
    latencies_ms = []
    cpu_usages = []

    for i in range(runs):
        cpu_percent_before = psutil.cpu_percent(interval=None)

        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()

        cpu_percent_after = psutil.cpu_percent(interval=None)
        latency_ms = (t1 - t0) * 1000.0

        latencies_ms.append(latency_ms)
        cpu_usages.append((cpu_percent_before + cpu_percent_after) / 2.0)

        print(f"  iter {i+1}/{runs}: {latency_ms:.2f} ms")

    lat = np.array(latencies_ms)
    cpu = np.array(cpu_usages)

    metrics = {
        "backend": "onnxruntime-cpu",
        "runs": runs,
        "img_size": img_size,
        "latency_ms": {
            "mean": float(lat.mean()),
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
        },
        "fps": float(1000.0 / lat.mean()) if lat.mean() > 0 else 0.0,
        "cpu_usage": {
            "mean": float(cpu.mean()),
            "min": float(cpu.min()),
            "max": float(cpu.max()),
        },
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[benchmarks] Tamamlandı. Sonuçlar: {out_path}")


if __name__ == "__main__":
    run_onnx_benchmark()
