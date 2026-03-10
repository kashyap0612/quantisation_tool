from transformers import AutoModelForImageClassification

model_name = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

model = AutoModelForImageClassification.from_pretrained(model_name)

model.save_pretrained("./plant_model")

print("Model downloaded successfully")