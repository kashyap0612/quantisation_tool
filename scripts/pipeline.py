from scripts.utils.model_inspector import (
    inspect_model,
    validate_vision_architecture,
)
from scripts.utils.download_model import download_model
from scripts.utils.benchmark import benchmark_model

from scripts.conversion.export_onnx import export_onnx
from scripts.conversion.verify_onnx import verify_onnx
from scripts.conversion.onnx_to_savedmodel import (
    convert_to_savedmodel,
)
from scripts.conversion.savedmodel_to_tflite import (
    convert_to_tflite,
)


# Step definitions with icons for UI display
PIPELINE_STEPS = [
    {"icon": "🔍", "label": "Inspecting model metadata"},
    {"icon": "🛡️", "label": "Validating architecture"},
    {"icon": "📥", "label": "Downloading model"},
    {"icon": "📦", "label": "Exporting to ONNX"},
    {"icon": "✅", "label": "Verifying ONNX graph"},
    {"icon": "🔄", "label": "Converting to SavedModel"},
    {"icon": "⚡", "label": "Quantizing to TFLite"},
    {"icon": "📊", "label": "Benchmarking accuracy"},
]


def _default_progress(step, total, icon, message):
    """Default callback: print to terminal."""
    print(f"\n[{step}/{total}] {icon} {message}")


def quantize_model(
    model_id: str,
    quantization: str = "fp16",
    on_progress=None
):
    """
    Complete quantization pipeline.

    Steps:
    1. Inspect model metadata
    2. Validate architecture (vision CNNs only)
    3. Download model
    4. Export to ONNX
    5. Verify ONNX
    6. Convert to TF SavedModel
    7. Convert to quantized TFLite
    8. Benchmark (cosine similarity)

    Args:
        model_id: Hugging Face model identifier
        quantization: 'fp16' or 'int8'
        on_progress: Optional callback(step, total, icon, message)
                     If None, prints to terminal.

    Returns dict with paths and benchmark results.
    """

    total = len(PIPELINE_STEPS)
    progress = on_progress or _default_progress

    # [1/8] Inspect
    step_info = PIPELINE_STEPS[0]
    progress(1, total, step_info["icon"], step_info["label"])
    model_info = inspect_model(model_id)

    # [2/8] Validate
    step_info = PIPELINE_STEPS[1]
    progress(2, total, step_info["icon"], step_info["label"])
    validate_vision_architecture(
        model_info["model_type"]
    )

    # [3/8] Download
    step_info = PIPELINE_STEPS[2]
    progress(3, total, step_info["icon"], step_info["label"])
    model_dir = download_model(model_id)

    # [4/8] Export ONNX
    step_info = PIPELINE_STEPS[3]
    progress(4, total, step_info["icon"], step_info["label"])
    onnx_path = export_onnx(model_dir)

    # [5/8] Verify ONNX
    step_info = PIPELINE_STEPS[4]
    progress(5, total, step_info["icon"], step_info["label"])
    verify_onnx(onnx_path)

    # [6/8] Convert to SavedModel
    step_info = PIPELINE_STEPS[5]
    progress(6, total, step_info["icon"], step_info["label"])
    saved_model_dir = convert_to_savedmodel(
        onnx_path
    )

    # [7/8] Quantize to TFLite
    step_info = PIPELINE_STEPS[6]
    progress(
        7, total, step_info["icon"],
        f"{step_info['label']} ({quantization.upper()})"
    )
    tflite_path = convert_to_tflite(
        saved_model_dir,
        quantization
    )

    # [8/8] Benchmark
    step_info = PIPELINE_STEPS[7]
    progress(8, total, step_info["icon"], step_info["label"])
    benchmark = benchmark_model(
        model_dir,
        tflite_path
    )

    return {
        "model_id": model_id,
        "model_info": model_info,
        "onnx_path": onnx_path,
        "saved_model_dir": saved_model_dir,
        "tflite_path": tflite_path,
        "benchmark": benchmark,
    }


if __name__ == "__main__":

    result = quantize_model(
        "google/mobilenet_v2_1.0_224",
        quantization="int8"
    )

    print("\nFinal Output:")
    print(f"TFLite: {result['tflite_path']}")
    print(
        f"Cosine Similarity: "
        f"{result['benchmark']['cosine_similarity']}"
    )
    print(
        f"Quality: {result['benchmark']['quality']}"
    )