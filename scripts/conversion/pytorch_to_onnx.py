import torch
from torchvision.models import mobilenet_v2
import os

# Ensure output directory exists
os.makedirs("models/onnx", exist_ok=True)

# Load pretrained MobileNetV2 (1000 classes - ImageNet)
model = mobilenet_v2(weights="IMAGENET1K_V1")
model.eval()

# Dummy input (batch_size=1, 3 channels, 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
onnx_path = "models/onnx/mobilenet_v2.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=13,
    do_constant_folding=True
)

print(f"ONNX export complete: {onnx_path}")