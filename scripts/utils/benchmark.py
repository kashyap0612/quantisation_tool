# scripts/utils/benchmark.py

from pathlib import Path
import json

import numpy as np
import torch
from transformers import AutoModelForImageClassification


def get_image_size(model_dir: str):
    """
    Read image size from preprocessor_config.json.
    Falls back to 224x224 if not found.
    """

    preprocessor_path = (
        Path(model_dir) / "preprocessor_config.json"
    )

    if not preprocessor_path.exists():
        return 224, 224

    with open(preprocessor_path, "r") as f:
        config = json.load(f)

    size = config.get("size", {})

    if isinstance(size, dict):
        height = size.get("height", 224)
        width = size.get("width", 224)
    else:
        height = width = int(size)

    return height, width


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
    Compute cosine similarity between two
    flattened output tensors.
    """

    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    dot = np.dot(a_flat, b_flat)

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def run_pytorch_inference(model_dir: str, dummy_input):
    """
    Run inference on the original PyTorch model.
    Returns numpy output.
    """

    model = AutoModelForImageClassification.from_pretrained(
        model_dir
    )
    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    logits = output.logits.numpy()

    return logits


def run_tflite_inference(tflite_path: str, dummy_input):
    """
    Run inference on the quantized TFLite model.
    Returns numpy output.

    Uses ai_edge_litert if available,
    falls back to tensorflow.lite.
    """

    # Convert from PyTorch NCHW to TFLite NHWC
    input_np = dummy_input.numpy().transpose(0, 2, 3, 1)

    try:
        from ai_edge_litert import interpreter as litert
        interp = litert.Interpreter(
            model_path=tflite_path
        )
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(
            model_path=tflite_path
        )

    interp.allocate_tensors()

    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Match the input dtype expected by the model
    expected_dtype = input_details[0]["dtype"]
    input_data = input_np.astype(expected_dtype)

    interp.set_tensor(
        input_details[0]["index"],
        input_data
    )

    interp.invoke()

    output = interp.get_tensor(
        output_details[0]["index"]
    )

    return output


def benchmark_model(
    model_dir: str,
    tflite_path: str
):
    """
    Compare PyTorch vs TFLite outputs using
    cosine similarity on a random dummy input.

    Returns dict with similarity score and
    quality assessment.
    """

    height, width = get_image_size(model_dir)

    # Fixed seed for reproducibility
    torch.manual_seed(42)

    dummy_input = torch.randn(1, 3, height, width)

    print("  Running PyTorch inference...")
    pytorch_output = run_pytorch_inference(
        model_dir, dummy_input
    )

    print("  Running TFLite inference...")
    tflite_output = run_tflite_inference(
        tflite_path, dummy_input
    )

    similarity = cosine_similarity(
        pytorch_output, tflite_output
    )

    # Quality assessment
    if similarity >= 0.95:
        quality = "Excellent"
    elif similarity >= 0.85:
        quality = "Good"
    elif similarity >= 0.70:
        quality = "Acceptable"
    else:
        quality = "Poor"

    result = {
        "cosine_similarity": round(similarity, 6),
        "quality": quality,
        "pytorch_output_shape": list(
            pytorch_output.shape
        ),
        "tflite_output_shape": list(
            tflite_output.shape
        ),
    }

    print(
        f"  Cosine Similarity: {result['cosine_similarity']}"
    )
    print(f"  Quality: {result['quality']}")

    return result


if __name__ == "__main__":

    model_dir = (
        "models/downloaded/"
        "google_mobilenet_v2_1.0_224"
    )
    tflite_path = (
        "models/tflite/"
        "google_mobilenet_v2_1.0_224_fp16.tflite"
    )

    result = benchmark_model(model_dir, tflite_path)

    print("\nBenchmark Results")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")
