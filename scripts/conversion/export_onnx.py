from pathlib import Path
import json

import torch
from transformers import AutoModelForImageClassification


def get_image_size(model_dir: str):
    """
    Read image size from preprocessor_config.json
    """

    preprocessor_path = Path(model_dir) / "preprocessor_config.json"

    with open(preprocessor_path, "r") as f:
        config = json.load(f)

    size = config.get("size", {})

    if isinstance(size, dict):
        height = size.get("height", 224)
        width = size.get("width", 224)
    else:
        height = width = int(size)

    return height, width


def export_onnx(model_dir: str):

    model_name = Path(model_dir).name

    output_dir = Path("models/onnx")
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / f"{model_name}.onnx"

    if onnx_path.exists():
        print(f"Using cached ONNX: {onnx_path}")
        return str(onnx_path)

    model = AutoModelForImageClassification.from_pretrained(
        model_dir
    )

    model.eval()

    height, width = get_image_size(model_dir)

    dummy_input = torch.randn(
        1,
        3,
        height,
        width
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=13,
        do_constant_folding=True
    )

    print(f"ONNX exported: {onnx_path}")

    return str(onnx_path)


if __name__ == "__main__":

    model_dir = (
        "models/downloaded/"
        "google_mobilenet_v2_1.0_224"
    )

    export_onnx(model_dir)