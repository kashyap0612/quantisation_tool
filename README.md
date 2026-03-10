Quantisation Tool

Pipeline:
PyTorch → ONNX → TFLite

Steps:
1. Download model
2. Convert to ONNX
3. Verify ONNX
4. Convert to TFLite
5. Run quantisation

Usage:
python scripts/download_model.py
python scripts/pytorch_to_onnx.py
python scripts/verify_onnx.py
python scripts/onnx_to_tflite.py