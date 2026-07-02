import tensorflow as tf

saved_model_dir = "models/tf_model"
tflite_quant_path = "models/tflite/mobilenet_v2_fp16.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# ✅ FLOAT16 quantization (stable)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(tflite_quant_path, "wb") as f:
    f.write(tflite_model)

print("FP16 model generated:", tflite_quant_path)