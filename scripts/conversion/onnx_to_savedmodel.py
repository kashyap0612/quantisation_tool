from pathlib import Path
import subprocess


def convert_to_savedmodel(onnx_path: str):

    model_name = Path(onnx_path).stem

    output_dir = Path("models/tf_model") / model_name

    if output_dir.exists():
        print(f"Using cached SavedModel: {output_dir}")
        return str(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "onnx2tf",
        "-i",
        onnx_path,
        "-o",
        str(output_dir)
    ]

    subprocess.run(cmd, check=True)

    print(f"SavedModel generated: {output_dir}")

    return str(output_dir)


if __name__ == "__main__":

    convert_to_savedmodel(
        "models/onnx/google_mobilenet_v2_1.0_224.onnx"
    )