import os, json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 5
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"
OUTPUT_DIR = "models/mobilenetv2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20, zoom_range=0.15,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.15, horizontal_flip=True,
                                   fill_mode="nearest")
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True)
val_gen = val_datagen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE,
                                          batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

idx_to_label = {v: k for k, v in train_gen.class_indices.items()}
print("Class indices:", idx_to_label)

base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(128,128,3)))
for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(len(idx_to_label), activation="softmax")(x)
model = Model(inputs=base.input, outputs=out)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_gen, steps_per_epoch=train_gen.samples // BATCH_SIZE,
          validation_data=val_gen, validation_steps=val_gen.samples // BATCH_SIZE,
          epochs=EPOCHS)

model.save(os.path.join(OUTPUT_DIR, "mask_detector_mbv2.keras"))
with open(os.path.join(OUTPUT_DIR, "class_labels.json"), "w") as f:
    json.dump(idx_to_label, f)
print("MobileNetV2 saved.")