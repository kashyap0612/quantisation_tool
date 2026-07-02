from transformers import AutoConfig


# Supported CNN vision architectures for quantization.
# These have been validated to convert cleanly through
# the PyTorch → ONNX → TF SavedModel → TFLite pipeline.
SUPPORTED_VISION_ARCHITECTURES = [
    "mobilenet_v1",
    "mobilenet_v2",
    "mobilenetv2",
    "resnet",
    "efficientnet",
    "convnext",
    "regnet",
    "squeezenet",
    "shufflenet",
    "shufflenet_v2",
    "densenet",
    "mnasnet",
    "googlenet",
    "inception",
    "vgg",
    "alexnet",
]


def inspect_model(model_id: str):
    """
    Fetch model metadata from Hugging Face
    without downloading model weights.
    """

    config = AutoConfig.from_pretrained(model_id)

    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "architectures": getattr(
            config,
            "architectures",
            []
        ) or [],
        "num_labels": getattr(config, "num_labels", None),
    }


def validate_vision_architecture(model_type: str):
    """
    Check if the model architecture is a
    supported vision CNN for quantization.

    Raises ValueError if not supported.
    Returns True if supported.
    """

    model_type_lower = model_type.lower()

    for arch in SUPPORTED_VISION_ARCHITECTURES:
        if arch in model_type_lower:
            return True

    supported_list = ", ".join(
        SUPPORTED_VISION_ARCHITECTURES
    )

    raise ValueError(
        f"Unsupported architecture: '{model_type}'. "
        f"This tool supports vision CNN models only. "
        f"Supported types: {supported_list}"
    )


if __name__ == "__main__":

    model_id = "google/mobilenet_v2_1.0_224"

    info = inspect_model(model_id)

    print("\nModel Information")
    print("-" * 40)

    for key, value in info.items():
        print(f"{key}: {value}")

    # Validate architecture
    try:
        validate_vision_architecture(info["model_type"])
        print("\n✓ Architecture is supported")
    except ValueError as e:
        print(f"\n✗ {e}")