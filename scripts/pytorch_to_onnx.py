from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
    output=".",
    task="image-classification"
)

print("ONNX export complete")