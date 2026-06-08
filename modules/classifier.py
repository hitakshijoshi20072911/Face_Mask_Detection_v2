import cv2, numpy as np, json, os
import tensorflow as tf

class MaskClassifier:
    def __init__(self, model_path, label_map_path, is_mobilenet=False, use_tflite=False):
        self.use_tflite = use_tflite
        with open(label_map_path, "r") as f:
            self.labels = json.load(f)   # dict str->str
        self.input_size = (128, 128)
        self.is_mobilenet = is_mobilenet

        if use_tflite:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, self.input_size)
        face = np.expand_dims(face, axis=0).astype(np.float32)
        if self.is_mobilenet:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            face = preprocess_input(face)
        else:
            face /= 255.0
        return face

    def evaluate_face(self, face_crop):
        """Return label, confidence (top prediction)."""
        processed = self.preprocess(face_crop)
        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], processed)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            preds = self.model.predict(processed, verbose=0)[0]
        idx = np.argmax(preds)
        return self.labels[str(idx)], preds[idx]

    def evaluate_face_raw(self, face_crop):
        """Return label, confidence, and full probability array."""
        processed = self.preprocess(face_crop)
        if self.use_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], processed)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            preds = self.model.predict(processed, verbose=0)[0]
        idx = np.argmax(preds)
        return self.labels[str(idx)], preds[idx], preds