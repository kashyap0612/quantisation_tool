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

---

## 📂 Project Structure

```
├── app.py                          # Gradio UI & Hugging Face Entrypoint
├── requirements.txt                # Cloud-optimized dependencies
├── scripts/
│   ├── pipeline.py                 # 8-step orchestrator
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