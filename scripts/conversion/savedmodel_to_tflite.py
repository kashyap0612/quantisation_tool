# scripts/conversion/savedmodel_to_tflite.py

from pathlib import Path
import tensorflow as tf


def convert_to_tflite(
    saved_model_dir: str,
    quantization: str = "fp16"
):
    """
    Convert TensorFlow SavedModel to TFLite.

    Supported quantization:
    - fp16
    - int8 (dynamic range)

    If conversion fails due to unsupported ops,
    automatically retries with SELECT_TF_OPS enabled.
    """

    model_name = Path(saved_model_dir).name

    output_dir = Path("models/tflite")
    output_dir.mkdir(parents=True, exist_ok=True)

    quantization = quantization.lower()

    if quantization not in ["fp16", "int8"]:
        raise ValueError(
            "quantization must be either 'fp16' or 'int8'"
        )

    tflite_path = (
        output_dir /
        f"{model_name}_{quantization}.tflite"
    )

    # Cache
    if tflite_path.exists():
        print(f"Using cached TFLite: {tflite_path}")
        return str(tflite_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_model_dir
    )

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    # FP16 Quantization
    if quantization == "fp16":
        converter.target_spec.supported_types = [
            tf.float16
        ]

    # Dynamic Range INT8 Quantization
    elif quantization == "int8":
        # tf.lite.Optimize.DEFAULT with no
        # representative dataset = dynamic range
        pass

    # First attempt: standard conversion
    try:
        tflite_model = converter.convert()

    except Exception as e:
        print(
            f"\n⚠ Standard conversion failed: {e}"
        )
        print(
            "  Retrying with SELECT_TF_OPS fallback..."
        )

        # Fallback: allow TF ops for unsupported layers
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

        try:
            tflite_model = converter.convert()
            print(
                "  ✓ Conversion succeeded with "
                "SELECT_TF_OPS (binary size may "
                "be larger)"
            )

        except Exception as fallback_err:
            raise RuntimeError(
                f"TFLite conversion failed even "
                f"with SELECT_TF_OPS: {fallback_err}"
            ) from fallback_err

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(
        f"{quantization.upper()} model generated: "
        f"{tflite_path}"
    )

    return str(tflite_path)


if __name__ == "__main__":

    saved_model_dir = (
        "models/tf_model/"
        "google_mobilenet_v2_1.0_224"
    )

    # FP16
    convert_to_tflite(
        saved_model_dir,
        quantization="fp16"
    )

    # INT8
    convert_to_tflite(
        saved_model_dir,
        quantization="int8"
    )