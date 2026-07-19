---
title: HF Vision Model Quantizer
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
---

# ⚡ HF Vision Model Quantizer

A cloud-native tool to **search, download, and quantize** Hugging Face vision models (CNNs) to TFLite format for bare-metal edge deployment.

👉 **[Try the Live Cloud Demo on Hugging Face Spaces!](https://huggingface.co/spaces/freeAAk48/HF_Model_Quantizer)**

> **Pipeline**: PyTorch → ONNX → TF SavedModel → Quantized TFLite

---

## 🛠️ Architecture & OS (COA) Discussion: Why Quantize for ARM?

This project bridges the gap between high-level Machine Learning (PyTorch) and low-level Computer Organization and Architecture (COA) for edge devices (specifically ARM-based OS environments like Android or Raspberry Pi). 

Running standard PyTorch FP32 models on ARM architectures is highly inefficient for several reasons:

1. **Cache Locality & Memory Bandwidth**: ARM processors have strictly constrained L1/L2 caches. A standard FP32 ResNet model might be 100MB+, causing constant cache misses and forcing the OS to fetch weights from main RAM, bottlenecking the memory bus and draining the battery. By compressing the model to **Dynamic INT8 (75% smaller)**, the entire computational graph can often fit directly into the CPU cache, drastically increasing memory bandwidth efficiency.
2. **ALU and SIMD Pipelines**: ARM chips feature specialized SIMD (Single Instruction, Multiple Data) execution units like **NEON**. NEON ALUs can process multiple INT8 operations in a single clock cycle, whereas FP32 math requires significantly more power and cycle overhead.
3. **Bare-metal OS Execution**: Embedded Operating Systems do not have the overhead to run massive Python interpreters and PyTorch binaries. Converting the final model to **TensorFlow Lite (`.tflite`)** allows the model to be executed directly via C++ bindings by the OS, stripping away virtualization overhead and allowing direct hardware acceleration (like the Android NNAPI or ARM Mali GPUs).

---

## ✨ Features

- 🔍 **Search** any model directly from the Hugging Face Hub.
- ☁️ **Cloud Native** — Fully automated CI/CD from GitHub to Hugging Face ZeroGPU Spaces.
- ⚙️ **FP16 & Dynamic INT8** hardware-aware quantization.
- 📊 **Cosine Similarity** benchmark — mathematically guarantees the quantized TFLite model matches the PyTorch original.
- 📥 **One-click download** of edge-ready binaries.

---

## 📊 Benchmark Results — MobileNetV2 1.0 224

> **Measurement environment**: Windows 11, AMD Ryzen CPU, TFLite CPU delegate (no NNAPI/NEON).  
> All numbers are **real measured values** from [`scripts/benchmark_comparison.py`](scripts/benchmark_comparison.py).  
> Latency = median of 50 inferences after 5 warm-up runs.

| Metric | FP32 (baseline) | FP16 | INT8 (dynamic range) |
|--------|:-----------:|:----:|:--------------------:|
| **Model size (MB)** | 13.34 | 6.70 | 3.61 |
| **Size reduction vs FP32** | 1× | ~2× | **~3.7×** |
| **Latency — median (ms/image)** | 70.40 | 56.68 | 1,935.95 ⚠️ |
| **Latency — p95 (ms/image)** | 76.07 | 60.89 | 2,332.95 ⚠️ |
| **Cosine similarity vs FP32** † | 1.000000 | ≥ 0.9997 | ≥ 0.9990 |
| **Quality classification** | Baseline | Excellent | Excellent |

> ⚠️ **INT8 latency caveat (important)**: Dynamic-range INT8 on a **Windows x86 CPU delegate** is slower than FP32 because TFLite must insert runtime dequantization kernels — there are no INT8-native SIMD intrinsics on this platform. The speedup only materialises on **ARM targets with NEON/NNAPI** (Android NDK, Raspberry Pi), which is the intended deployment environment. On-device ARM, expected latency is **~15–30 ms** (INT8) vs **~80–120 ms** (FP32).

> † Cosine similarity is measured between PyTorch FP32 logits and TFLite quantized logits on the **same input tensor** — not between different source checkpoints. Computed by [`scripts/utils/benchmark.py`](scripts/utils/benchmark.py).

### Understanding the FP16 → INT8 accuracy trade-off

FP16 quantization is near-lossless at the logit level — values retain full dynamic range at halved precision. INT8 dynamic-range quantization introduces two compounding error sources:

1. **Activation range clipping** — outlier activations outside the calibrated min/max range are saturated to INT8 bounds (−128 / +127).
2. **Scale factor rounding** — each tensor's float values are divided by a per-tensor scale and rounded; accumulated rounding error propagates layer-by-layer.

The net result is a small logit perturbation (cosine similarity ≥ 0.999) that translates to approximately **1–2% top-1 accuracy drop** in standard ImageNet evaluation for MobileNetV2-class models — an acceptable trade-off for the **3.7× size reduction** and the significant ARM inference speedup.

---

## 🏗️ The 8-Step Pipeline

```
[1] Inspect model metadata from Hugging Face
[2] Validate architecture (vision CNNs only)
[3] Download model weights & processor
[4] Export to ONNX (opset 13, dynamic batch)
[5] Verify ONNX graph integrity
[6] Convert ONNX → TF SavedModel (via onnx2tf)
[7] Quantize SavedModel → TFLite (FP16 / INT8)
[8] Benchmark accuracy (Cosine Similarity)
```

---

## 🚀 Running Locally

### Prerequisites

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Run the UI (Gradio)

```bash
python app.py
```

### Run the Benchmark

```bash
python scripts/benchmark_comparison.py
```

---

## 📂 Project Structure

```
├── app.py                          # Gradio UI & Hugging Face Entrypoint
├── requirements.txt                # Cloud-optimized dependencies
├── benchmark_results.json          # Latest measured benchmark numbers
├── scripts/
│   ├── pipeline.py                 # 8-step orchestrator
│   ├── benchmark_comparison.py     # Size / latency / RAM comparator
│   ├── utils/
│   │   ├── hf_search.py            # HF Hub search
│   │   ├── model_inspector.py      # Architecture validation
│   │   ├── download_model.py       # Model downloader
│   │   └── benchmark.py            # Cosine similarity verifier
│   └── conversion/
│       ├── export_onnx.py          # PyTorch → ONNX
│       ├── verify_onnx.py          # ONNX validation
│       ├── onnx_to_savedmodel.py   # ONNX → TF SavedModel
│       └── savedmodel_to_tflite.py # SavedModel → TFLite
```

---

## 📄 License

MIT