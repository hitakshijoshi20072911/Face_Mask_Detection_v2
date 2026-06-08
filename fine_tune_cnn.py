import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 5

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
train_gen = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

model = load_model("models/custom_cnn/mask_detector_cnn.keras")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # low learning rate
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_gen,
          steps_per_epoch=train_gen.samples // BATCH_SIZE,
          epochs=EPOCHS)

model.save("models/custom_cnn/mask_detector_cnn.keras")
print("Fine‑tuned model saved.")