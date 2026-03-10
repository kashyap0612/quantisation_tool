from onnx2tf import convert

convert(
    input_onnx_file_path="plant_disease.onnx",
    batch_size=1,
    param_replacement_file="saved_model/plant_disease_auto.json"
)

print("TFLite model generated")