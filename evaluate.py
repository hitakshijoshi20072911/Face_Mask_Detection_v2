import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Choose model: "custom_cnn" or "mobilenetv2"
MODEL_TYPE = "mobilenetv2"

if MODEL_TYPE == "custom_cnn":
    MODEL_FILE = "models/custom_cnn/mask_detector_cnn.keras"
    LABEL_FILE = "models/custom_cnn/class_labels.json"
elif MODEL_TYPE == "mobilenetv2":
    MODEL_FILE = "models/mobilenetv2/mask_detector_mbv2.keras"
    LABEL_FILE = "models/mobilenetv2/class_labels.json"
else:
    raise ValueError("Invalid MODEL_TYPE")

with open(LABEL_FILE, "r") as f:
    idx_to_label = json.load(f)
class_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

if "mobilenet" in MODEL_TYPE:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
else:
    test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory("dataset/test", target_size=(128,128),
                                            batch_size=32, class_mode="categorical", shuffle=False)

model = load_model(MODEL_FILE)
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix - {MODEL_TYPE}")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()