import onnx

model = onnx.load("plant_disease.onnx")
onnx.checker.check_model(model)

print("ONNX model is valid")