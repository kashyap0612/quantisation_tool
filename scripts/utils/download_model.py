from pathlib import Path

from transformers import (
    AutoModel,
    AutoModelForImageClassification,
    AutoConfig,
    AutoProcessor,
)


def download_model(model_id: str):
    """
    Download a Hugging Face model and cache it locally.
    If already downloaded, reuse existing files.
    """

    model_dir = Path(
        "models/downloaded/" +
        model_id.replace("/", "_")
    )

    # Reuse cached model
    if model_dir.exists():
        print(f"Using cached model: {model_dir}")
        return str(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(model_id)

    try:
        try:
            model = AutoModelForImageClassification.from_pretrained(
                model_id
            )
        except Exception:
            model = AutoModel.from_pretrained(
                model_id
            )

        model.save_pretrained(model_dir)

    except Exception as e:
        raise RuntimeError(
            f"Failed to download model: {e}"
        )

    # Processor is optional
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        processor.save_pretrained(model_dir)
    except Exception:
        pass

    print(f"Model saved to: {model_dir}")

    return str(model_dir)


if __name__ == "__main__":

    model_id = "google/mobilenet_v2_1.0_224"

    path = download_model(model_id)

    print("\nDownloaded Path:")
    print(path)