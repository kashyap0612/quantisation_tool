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


def quantize_model(
    model_id: str,
    quantization: str = "fp16"
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

    Returns dict with paths and benchmark results.
    """

    print("\n[1/8] Inspecting model...")
    model_info = inspect_model(model_id)

    print("Architecture:",
          model_info["architectures"])

    print("\n[2/8] Validating architecture...")
    validate_vision_architecture(
        model_info["model_type"]
    )
    print("✓ Architecture supported")

    print("\n[3/8] Downloading model...")
    model_dir = download_model(model_id)

    print("\n[4/8] Exporting ONNX...")
    onnx_path = export_onnx(model_dir)

    print("\n[5/8] Verifying ONNX...")
    verify_onnx(onnx_path)

    print("\n[6/8] Converting to SavedModel...")
    saved_model_dir = convert_to_savedmodel(
        onnx_path
    )

    print(
        f"\n[7/8] Generating {quantization.upper()} "
        f"TFLite..."
    )

    tflite_path = convert_to_tflite(
        saved_model_dir,
        quantization
    )

    print("\n[8/8] Benchmarking accuracy...")
    benchmark = benchmark_model(
        model_dir,
        tflite_path
    )

    print("\n✓ Pipeline Complete")

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