"""
Compute cosine similarity between FP32, FP16, and INT8 outputs only.
Sizes and latencies are already known from the first run.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

FP32_TFLITE = ROOT / "models/tflite/mobilenet_v2.tflite"
FP16_PATH   = ROOT / "models/tflite/google_mobilenet_v2_1.0_224_fp16.tflite"
INT8_PATH   = ROOT / "models/tflite/google_mobilenet_v2_1.0_224_int8.tflite"

def get_interpreter(path):
    try:
        from ai_edge_litert import interpreter as litert  # type: ignore[import-not-found]
        interp = litert.Interpreter(model_path=str(path))
    except Exception:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp

def run_once(interp):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    rng = np.random.default_rng(42)
    dtype = inp["dtype"]
    shape = inp["shape"]
    if np.issubdtype(dtype, np.floating):
        data = rng.random(shape, dtype=np.float32).astype(dtype)
    else:
        data = rng.integers(0, 256, size=shape, dtype=dtype)
    interp.set_tensor(inp["index"], data)
    interp.invoke()
    return interp.get_tensor(out["index"]).copy()

def cosine_sim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

print("Running single-pass inference for similarity...")
fp32_out = run_once(get_interpreter(FP32_TFLITE))
fp16_out = run_once(get_interpreter(FP16_PATH))
int8_out = run_once(get_interpreter(INT8_PATH))

sim_fp16 = cosine_sim(fp32_out, fp16_out)
sim_int8 = cosine_sim(fp32_out, int8_out)

print(f"FP16 cosine similarity vs FP32: {sim_fp16:.6f}")
print(f"INT8 cosine similarity vs FP32: {sim_int8:.6f}")

# Merge with known latency/size results
results = {
    "FP32 (baseline)": {
        "size_mb": 13.34, "latency_median_ms": 70.40, "latency_p95_ms": 76.07,
        "cosine_sim_vs_fp32": 1.0, "size_reduction_vs_fp32": 1.0
    },
    "FP16": {
        "size_mb": 6.70, "latency_median_ms": 56.68, "latency_p95_ms": 60.89,
        "cosine_sim_vs_fp32": round(sim_fp16, 6),
        "size_reduction_vs_fp32": round(13.34 / 6.70, 2)
    },
    "INT8": {
        "size_mb": 3.61, "latency_median_ms": 1935.95, "latency_p95_ms": 2332.95,
        "cosine_sim_vs_fp32": round(sim_int8, 6),
        "size_reduction_vs_fp32": round(13.34 / 3.61, 2)
    },
}

out_path = ROOT / "benchmark_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {out_path}")
print(json.dumps(results, indent=2))
