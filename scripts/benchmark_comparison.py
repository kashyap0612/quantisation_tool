"""
benchmark_comparison.py
-----------------------
Measures model size, inference latency (ms/image) and peak RAM
for the FP32, FP16, and INT8 variants of MobileNetV2
using the already-converted TFLite files.

Results are saved to: benchmark_results.json
Run from the project root:
    python scripts/benchmark_comparison.py
"""

import os
import sys
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

FP32_TFLITE = ROOT / "models/tflite/mobilenet_v2.tflite"
FP16_PATH   = ROOT / "models/tflite/google_mobilenet_v2_1.0_224_fp16.tflite"
INT8_PATH   = ROOT / "models/tflite/google_mobilenet_v2_1.0_224_int8.tflite"

N_WARMUP = 5
N_RUNS   = 50

# ── helpers ────────────────────────────────────────────────────────────

def file_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 ** 2), 2)


def make_dummy_input(interp):
    inp   = interp.get_input_details()[0]
    shape = inp["shape"]
    dtype = inp["dtype"]
    rng   = np.random.default_rng(42)
    if np.issubdtype(dtype, np.floating):
        arr = rng.random(shape, dtype=np.float32).astype(dtype)
    else:
        arr = rng.integers(0, 256, size=shape, dtype=dtype)
    return arr


def get_interpreter(model_path: Path):
    try:
        from ai_edge_litert import interpreter as litert
        interp = litert.Interpreter(model_path=str(model_path))
    except (ImportError, Exception):
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    return interp


def benchmark_tflite(model_path: Path):
    interp  = get_interpreter(model_path)
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    dummy   = make_dummy_input(interp)

    # warm-up
    for _ in range(N_WARMUP):
        interp.set_tensor(inp_det[0]["index"], dummy)
        interp.invoke()

    # timed runs
    latencies = []
    tracemalloc.start()
    for _ in range(N_RUNS):
        interp.set_tensor(inp_det[0]["index"], dummy)
        t0 = time.perf_counter()
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    output = interp.get_tensor(out_det[0]["index"]).copy()

    return {
        "latency_median_ms": round(float(np.median(latencies)), 2),
        "latency_p95_ms":    round(float(np.percentile(latencies, 95)), 2),
        "peak_ram_mb":       round(peak / (1024 ** 2), 2),
        "output":            output,
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    # Truncate to min length to handle off-by-one class count differences
    # (e.g. INT8 conversion adds a background class → 1001 vs 1000)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TFLite Quantization Benchmark — MobileNetV2 1.0 224")
    print("=" * 60)

    variants = {
        "FP32 (baseline)": FP32_TFLITE,
        "FP16":            FP16_PATH,
        "INT8":            INT8_PATH,
    }

    for name, path in variants.items():
        if not path.exists():
            print(f"\nERROR: {name} file not found → {path}")
            sys.exit(1)

    results = {}
    outputs = {}

    for name, path in variants.items():
        print(f"\nBenchmarking {name}  ({file_mb(path)} MB) ...")
        r = benchmark_tflite(path)
        outputs[name] = r.pop("output")
        results[name] = {
            "size_mb":           file_mb(path),
            "latency_median_ms": r["latency_median_ms"],
            "latency_p95_ms":    r["latency_p95_ms"],
            "peak_ram_mb":       r["peak_ram_mb"],
        }
        print(f"   size:           {results[name]['size_mb']} MB")
        print(f"   latency median: {results[name]['latency_median_ms']} ms")
        print(f"   latency p95:    {results[name]['latency_p95_ms']} ms")
        print(f"   peak RAM:       {results[name]['peak_ram_mb']} MB")

    # Cosine similarity vs FP32 baseline
    fp32_out = outputs["FP32 (baseline)"]
    for name in ["FP16", "INT8"]:
        sim = cosine_similarity(fp32_out, outputs[name])
        results[name]["cosine_sim_vs_fp32"] = round(sim, 6)
    results["FP32 (baseline)"]["cosine_sim_vs_fp32"] = 1.0

    # Compression ratios
    fp32_size = results["FP32 (baseline)"]["size_mb"]
    for name in ["FP16", "INT8"]:
        results[name]["size_reduction_vs_fp32"] = round(fp32_size / results[name]["size_mb"], 2)
    results["FP32 (baseline)"]["size_reduction_vs_fp32"] = 1.0

    # Save
    out_path = ROOT / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    hdr = f"{'Metric':<30} {'FP32':>14} {'FP16':>14} {'INT8':>14}"
    print(hdr)
    print("-" * len(hdr))

    rows = [
        ("Size (MB)",              "size_mb"),
        ("Latency median (ms)",    "latency_median_ms"),
        ("Latency p95 (ms)",       "latency_p95_ms"),
        ("Peak RAM (MB)",          "peak_ram_mb"),
        ("Cosine sim vs FP32",     "cosine_sim_vs_fp32"),
        ("Size reduction x",       "size_reduction_vs_fp32"),
    ]

    names = ["FP32 (baseline)", "FP16", "INT8"]
    for label, key in rows:
        row = f"{label:<30}"
        for n in names:
            val = results[n][key]
            row += f" {val:>14}"
        print(row)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
