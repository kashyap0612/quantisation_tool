from pathlib import Path
import onnx


def verify_onnx(onnx_path: str):

    model = onnx.load(onnx_path)

    onnx.checker.check_model(model)

    print(f"ONNX model is valid: {onnx_path}")

    return True


if __name__ == "__main__":

    onnx_path = (
        "models/onnx/"
        "google_mobilenet_v2_1.0_224.onnx"
    )

    verify_onnx(onnx_path)