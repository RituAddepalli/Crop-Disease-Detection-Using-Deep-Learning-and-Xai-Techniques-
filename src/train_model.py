# current mobilenetv2 model with 97% accuracy 
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 20

DATASET_DIR = r"C:\projects\Crop Disease Detection-clg\dataset\train_dataset"
SAVE_PATH = r"C:\projects\Crop Disease Detection-clg\saved_models\best_crop_disease_model.keras"

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
print(f"\nClasses ({len(class_names)}):", class_names)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.15)
], name="augmentation")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y)).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -----------------------------
# BUILD MODEL (MobileNetV2)
# -----------------------------
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=x
)

base_model.trainable = False

# -----------------------------
# CLASSIFIER HEAD
# -----------------------------
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.BatchNormalization()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# -----------------------------
# COMPILE PHASE 1
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =============================
# PHASE 1 — Train Head
# =============================
print("\nPhase 1 — Training classifier head")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# =============================
# PHASE 2 — Fine Tuning
# =============================
print("\nPhase 2 — Fine tuning MobileNetV2")

# Unfreeze backbone
for layer in model.layers[:120]:
    layer.trainable = False

for layer in model.layers[120:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

reduce_lr2 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
    callbacks=[early_stop2, reduce_lr2, checkpoint]
)

print(f"\nBest model saved at: {SAVE_PATH}")

# -----------------------------
# PLOT TRAINING CURVES
# -----------------------------
acc1 = history1.history["accuracy"]
vacc1 = history1.history["val_accuracy"]

acc2 = history2.history["accuracy"]
vacc2 = history2.history["val_accuracy"]

all_acc = acc1 + acc2
all_vacc = vacc1 + vacc2

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(all_acc,label="Train")
plt.plot(all_vacc,label="Validation")
plt.axvline(x=len(acc1)-1,color="gray",linestyle="--")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
loss1 = history1.history["loss"]
vloss1 = history1.history["val_loss"]

loss2 = history2.history["loss"]
vloss2 = history2.history["val_loss"]

plt.plot(loss1+loss2,label="Train")
plt.plot(vloss1+vloss2,label="Validation")
plt.axvline(x=len(loss1)-1,color="gray",linestyle="--")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()




# with MobileNetV2
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # MobileNetV2 MODEL
# # -----------------------------

# inputs = tf.keras.Input(shape=(224,224,3))

# # Preprocess for MobileNet
# x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# base_model = tf.keras.applications.MobileNetV2(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=x
# )

# # Freeze base model
# base_model.trainable = False

# x = layers.GlobalAveragePooling2D()(base_model.output)

# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_mobilenetv2_model.keras")

# print("✔ MobileNetV2 model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("MobileNetV2 Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train","Validation"])

# plt.show()






#  # with EfficientNetB0
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # EfficientNet MODEL
# # -----------------------------

# inputs = tf.keras.Input(shape=(224,224,3))

# # Preprocess for EfficientNet
# x = tf.keras.applications.efficientnet.preprocess_input(inputs)

# base_model = tf.keras.applications.EfficientNetB0(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=x
# )

# # Freeze base model
# base_model.trainable = False

# x = layers.GlobalAveragePooling2D()(base_model.output)

# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_efficientnet_model.keras")

# print("✔ EfficientNet model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("EfficientNetB0 Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train","Validation"])

# plt.show()





# # with resnet50
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # RESNET50 MODEL
# # -----------------------------

# inputs = tf.keras.Input(shape=(224,224,3))

# # Preprocess for ResNet
# x = tf.keras.applications.resnet50.preprocess_input(inputs)

# base_model = tf.keras.applications.ResNet50(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=x
# )

# # Freeze base model
# base_model.trainable = False

# x = layers.GlobalAveragePooling2D()(base_model.output)

# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_resnet50_model")

# print("✔ ResNet50 model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("ResNet50 Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train","Validation"])

# plt.show()




# # with cnn
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers, models

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # CNN MODEL
# # -----------------------------
# model = models.Sequential([

#     layers.Rescaling(1./255, input_shape=(224,224,3)),

#     layers.Conv2D(32, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(64, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(128, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(256, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.GlobalAveragePooling2D(),

#     layers.Dropout(0.3),

#     layers.Dense(len(class_names), activation="softmax")
# ])

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_cnn_model.keras")
# print("✔ CNN model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("CNN Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train", "Validation"])

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
