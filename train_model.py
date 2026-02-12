from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# -----------------------------
# LOAD DATASET
# -----------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# BUILD MODEL (MobileNetV2)
# -----------------------------
inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=x
)

# Freeze base model (important)
base_model.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# -----------------------------
# COMPILE
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
checkpoint = ModelCheckpoint(
    "best_crop_disease_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# TRAIN
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# SAVE FINAL MODEL
# -----------------------------
model.save("crop_disease_mobilenetv2_FINAL.keras")
print("✔ Final model saved.")

# -----------------------------
# PLOT GRAPHS
# -----------------------------
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])

plt.tight_layout()
plt.show()


























# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers, models

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR, validation_split=0.2, subset="training",
#     seed=42, image_size=(IMG_SIZE, IMG_SIZE), batch_size=32)

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR, validation_split=0.2, subset="validation",
#     seed=42, image_size=(IMG_SIZE, IMG_SIZE), batch_size=32)

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # ------------------------------

# # FIXED: BUILD FUNCTIONAL MODEL
# # ------------------------------
# inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")

# # preprocessing should NOT include separate InputLayers
# x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# base = tf.keras.applications.MobileNetV2(
#     include_top=False, weights="imagenet", input_tensor=x)

# x = layers.GlobalAveragePooling2D()(base.output)
# x = layers.Dropout(0.3)(x)
# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)
# model.compile(optimizer="adam",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])

# model.summary()

# model.fit(train_ds, validation_data=val_ds, epochs=20)

# # this model WILL work with Grad-CAM
# model.save("crop_disease_mobilenetv2_FIXED_FINAL.keras")
# print("✔ Saved fixed model")
# plt.figure()
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title("Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(['Train', 'Validation'])
# plt.show()






















# import tensorflow as tf
# from tensorflow.keras import layers

# IMG_SIZE = 224
# BATCH_SIZE = 32
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # ------------------------------
# # Load Dataset
# # ------------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE
# )

# class_names = train_ds.class_names
# print("Detected Classes:", class_names)

# train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
# val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# # ------------------------------
# # Functional Model (Correct fix)
# # ------------------------------
# inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

# # rescale
# x = layers.Rescaling(1./255)(inputs)

# # MobileNetV2 with *connected input*
# base_model = tf.keras.applications.MobileNetV2(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=x   # <<< THIS FIXES THE GRAPH
# )

# base_model.trainable = False

# # Classification head
# x = layers.GlobalAveragePooling2D()(base_model.output)
# x = layers.Dropout(0.3)(x)
# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

# # compile
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# model.fit(train_ds, validation_data=val_ds, epochs=1)

# # Save model
# model.save("crop_disease_mobilenetv2_fixed.keras")

# print("✔ Model saved: crop_disease_mobilenetv2_fixed.keras")
