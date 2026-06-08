import os, json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 10
TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"
OUTPUT_DIR = "models/custom_cnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20, zoom_range=0.15,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.15, horizontal_flip=True,
                                   fill_mode="nearest")
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True)
val_gen = val_datagen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE,
                                          batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

idx_to_label = {v: k for k, v in train_gen.class_indices.items()}
print("Class indices:", idx_to_label)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(idx_to_label), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_gen, steps_per_epoch=train_gen.samples // BATCH_SIZE,
          validation_data=val_gen, validation_steps=val_gen.samples // BATCH_SIZE,
          epochs=EPOCHS)

model.save(os.path.join(OUTPUT_DIR, "mask_detector_cnn.keras"))
with open(os.path.join(OUTPUT_DIR, "class_labels.json"), "w") as f:
    json.dump(idx_to_label, f)
print("Custom CNN saved.")