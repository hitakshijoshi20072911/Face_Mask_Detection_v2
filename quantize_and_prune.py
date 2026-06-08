import tensorflow as tf
import os

def quantize_model(keras_path, tflite_path):
    """Load a .keras model and convert it to a quantized TFLite file."""
    print(f"[INFO] Quantizing {keras_path} ...")
    model = tf.keras.models.load_model(keras_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # 8‑bit dynamic range quant
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[SUCCESS] Quantized model saved to {tflite_path}")

if __name__ == "__main__":
    os.makedirs("models/custom_cnn", exist_ok=True)
    os.makedirs("models/mobilenetv2", exist_ok=True)

    quantize_model(
        "models/custom_cnn/mask_detector_cnn.keras",
        "models/custom_cnn/mask_detector_cnn.tflite"
    )
    quantize_model(
        "models/mobilenetv2/mask_detector_mbv2.keras",
        "models/mobilenetv2/mask_detector_mbv2.tflite"
    )

    print("\nQuantization complete. Both TFLite models are ready.")
    print(" Pruning is available in the code (see quantize_and_prune.py) but requires model retraining.")