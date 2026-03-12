import tensorflow as tf
import numpy as np
import os
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ eval.py is in src/ → go one level up to reach project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ model is in saved_models/
model = tf.keras.models.load_model(
    os.path.join(ROOT_DIR, "saved_models", "best_crop_disease_model.keras")
)

IMG_SIZE = 224

# ✅ test dataset
TEST_DIR = os.path.join(ROOT_DIR, "dataset", "test_dataset")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    shuffle=False
)

class_names = test_ds.class_names

# -------------------------
# Evaluate Accuracy
# -------------------------
loss, accuracy = model.evaluate(test_ds)

# -------------------------
# Predictions
# -------------------------
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# -------------------------
# Classification Report
# -------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# -------------------------
# Overall Test Accuracy
# -------------------------
print("\nOverall Test Accuracy:")
print(f"{accuracy:.4f}")

# -------------------------
# Per-Crop Accuracy
# -------------------------
print("\nPer-Crop Accuracy:\n")

crop_classes = {
    "Tomato": [i for i, c in enumerate(class_names) if c.startswith("Tomato")],
    "Potato": [i for i, c in enumerate(class_names) if c.startswith("Potato")],
    "Corn":   [i for i, c in enumerate(class_names) if c.startswith("Corn")]
}

for crop, indices in crop_classes.items():

    mask = np.isin(y_true, indices)

    crop_true = y_true[mask]
    crop_pred = y_pred[mask]

    correct = np.sum(crop_true == crop_pred)
    total = len(crop_true)

    acc = (correct / total) * 100

    print(f"{crop} Accuracy: {correct}/{total} = {acc:.2f}%")

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(16,12))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="magma",
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size":8}
)

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix", fontsize=16)

plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
plt.show()

# -------------------------
# Latency Test
# -------------------------
sample_batch = next(iter(test_ds))[0]

start = time.time()
model.predict(sample_batch)
end = time.time()

latency = (end - start) / len(sample_batch)

print("\nAverage Inference Time per Image (seconds):")
print(latency)


# # -----------------------------
# # SUPPRESS TENSORFLOW LOGS
# # -----------------------------
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # -----------------------------
# # IMPORT LIBRARIES
# # -----------------------------
# import tensorflow as tf
# import numpy as np
# import time
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # -----------------------------
# # CONFIG
# # -----------------------------
# IMG_SIZE = 224
# TEST_DIR = r"C:\projects\cropdiseasedetection\dataset\test_dataset"
# MODEL_PATH = r"C:\projects\cropdiseasedetection\src\crop_disease_mobilenetv2_FIXED_FINAL.keras"

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# print("\nLoading model...")
# model = tf.keras.models.load_model(MODEL_PATH)
# print("Model loaded successfully\n")

# # -----------------------------
# # LOAD TEST DATASET
# # -----------------------------
# print("Loading test dataset...")

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32,
#     shuffle=False
# )

# print("Dataset loaded successfully\n")

# class_names = test_ds.class_names
# print("Classes detected:", class_names)
# print("Total Classes:", len(class_names), "\n")

# # -----------------------------
# # EVALUATE MODEL
# # -----------------------------
# print("Evaluating model...\n")

# loss, accuracy = model.evaluate(test_ds, verbose=1)

# print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
# print("Test Loss:", loss)

# # -----------------------------
# # PREDICTIONS
# # -----------------------------
# print("\nGenerating predictions...")

# y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)

# y_pred_probs = model.predict(test_ds, verbose=1)
# y_pred = np.argmax(y_pred_probs, axis=1)

# print("Predictions completed\n")

# # -----------------------------
# # CLASSIFICATION REPORT
# # -----------------------------
# print("Classification Report:\n")

# print(
#     classification_report(
#         y_true,
#         y_pred,
#         target_names=class_names
#     )
# )

# # -----------------------------
# # PER-CROP ACCURACY
# # -----------------------------
# print("\nPer-Crop Accuracy:\n")

# crop_classes = {
#     "Tomato": [i for i, c in enumerate(class_names) if c.startswith("Tomato")],
#     "Corn": [i for i, c in enumerate(class_names) if c.startswith("Corn")],
#     "Potato": [i for i, c in enumerate(class_names) if c.startswith("Potato")],
# }

# for crop, indices in crop_classes.items():

#     mask = np.isin(y_true, indices)

#     crop_true = y_true[mask]
#     crop_pred = y_pred[mask]

#     correct = np.sum(crop_true == crop_pred)
#     total = len(crop_true)

#     acc = (correct / total) * 100

#     print(f"{crop}: {correct}/{total} = {acc:.2f}%")

# # -----------------------------
# # CONFUSION MATRIX
# # -----------------------------
# print("\nGenerating confusion matrix...")

# cm = confusion_matrix(y_true, y_pred)

# plt.figure(figsize=(12, 10))

# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     xticklabels=class_names,
#     yticklabels=class_names,
#     cmap="Blues"
# )

# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")

# plt.xticks(rotation=90)
# plt.yticks(rotation=0)

# plt.tight_layout()
# plt.show()

# # -----------------------------
# # LATENCY TEST
# # -----------------------------
# print("\nMeasuring inference latency...")

# sample_image = next(iter(test_ds))[0][0:1]

# start = time.time()
# model.predict(sample_image, verbose=0)
# end = time.time()

# latency = end - start

# print("\nAverage Inference Time per Image:", round(latency, 4), "seconds")

# print("\nEvaluation completed successfully.")









# got error
# # mobilenetv2-96 Accuracy
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# import tensorflow as tf
# import numpy as np
# import os
# import time
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load model
# model = tf.keras.models.load_model(r"C:\projects\cropdiseasedetection\src\crop_disease_mobilenetv2_FIXED_FINAL.keras")

# # model = tf.keras.models.load_model("best_crop_disease_model.keras")


# IMG_SIZE = 224
# TEST_DIR = r"C:\projects\cropdiseasedetection\dataset\test_dataset"

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32,
#     shuffle=False
# )

# class_names = test_ds.class_names

# # -------------------------
# # Evaluate Accuracy
# # -------------------------
# loss, accuracy = model.evaluate(test_ds)
# print("Test Accuracy:", accuracy)

# # -------------------------
# # Predictions
# # -------------------------
# y_true = np.concatenate([y for x, y in test_ds], axis=0)
# y_pred_probs = model.predict(test_ds)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # -------------------------
# # Classification Report
# # -------------------------
# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))

# # -------------------------
# # Per-Crop Accuracy
# # -------------------------
# crop_classes = {
#     "Tomato": [i for i, c in enumerate(class_names) if c.startswith("Tomato")],
#     "Corn":   [i for i, c in enumerate(class_names) if c.startswith("Corn")],
#     "Potato": [i for i, c in enumerate(class_names) if c.startswith("Potato")],
# }

# print("\nPer-Crop Accuracy:\n")
# for crop, indices in crop_classes.items():
#     mask = np.isin(y_true, indices)
#     crop_true = y_true[mask]
#     crop_pred = y_pred[mask]
#     correct = np.sum(crop_true == crop_pred)
#     total = len(crop_true)
#     print(f"{crop}: {correct}/{total} = {correct/total*100:.2f}%")

# # -------------------------
# # Confusion Matrix
# # -------------------------
# cm = confusion_matrix(y_true, y_pred)

# plt.figure()
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # -------------------------
# # Latency Test
# # -------------------------
# sample_batch = next(iter(test_ds))[0]

# start = time.time()
# model.predict(sample_batch)
# end = time.time()

# latency = (end - start) / len(sample_batch)
# print("Average Inference Time per Image (seconds):", latency)


# # mobilenetv2-96 Accuracy
# import tensorflow as tf
# import numpy as np
# import os
# import time
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load model
# model = tf.keras.models.load_model("best_crop_disease_model.keras")

# IMG_SIZE = 224
# TEST_DIR = r"C:\projects\cropdiseasedetection\test_dataset"

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32,
#     shuffle=False
# )

# class_names = test_ds.class_names

# # -------------------------
# # Evaluate Accuracy
# # -------------------------
# loss, accuracy = model.evaluate(test_ds)
# print("Test Accuracy:", accuracy)

# # -------------------------
# # Predictions
# # -------------------------
# y_true = np.concatenate([y for x, y in test_ds], axis=0)
# y_pred_probs = model.predict(test_ds)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # -------------------------
# # Classification Report
# # -------------------------
# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))

# # -------------------------
# # Confusion Matrix
# # -------------------------
# cm = confusion_matrix(y_true, y_pred)

# plt.figure()
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # -------------------------
# # Latency Test
# # -------------------------
# sample_batch = next(iter(test_ds))[0]

# start = time.time()
# model.predict(sample_batch)
# end = time.time()

# latency = (end - start) / len(sample_batch)
# print("Average Inference Time per Image (seconds):", latency)








# # Mobilenetv2
# import tensorflow as tf
# import numpy as np
# import os
# import time
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load model
# model = tf.keras.models.load_model("crop_disease_mobilenetv2_model.keras")

# IMG_SIZE = 224
# TEST_DIR = r"C:\projects\cropdiseasedetection\test_dataset"

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32,
#     shuffle=False
# )

# class_names = test_ds.class_names

# # -------------------------
# # Evaluate Accuracy
# # -------------------------
# loss, accuracy = model.evaluate(test_ds)
# print("Test Accuracy:", accuracy)

# # -------------------------
# # Predictions
# # -------------------------
# y_true = np.concatenate([y for x, y in test_ds], axis=0)
# y_pred_probs = model.predict(test_ds)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # -------------------------
# # Classification Report
# # -------------------------
# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))

# # -------------------------
# # Confusion Matrix
# # -------------------------
# cm = confusion_matrix(y_true, y_pred)

# plt.figure()
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # -------------------------
# # Latency Test
# # -------------------------
# sample_batch = next(iter(test_ds))[0]

# start = time.time()
# model.predict(sample_batch)
# end = time.time()

# latency = (end - start) / len(sample_batch)
# print("Average Inference Time per Image (seconds):", latency)












# efficientnet

# import tensorflow as tf
# import numpy as np
# import os
# import time
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load model
# model = tf.keras.models.load_model("crop_disease_efficientnet_model.keras")

# IMG_SIZE = 224
# TEST_DIR = r"C:\projects\cropdiseasedetection\test_dataset"

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32,
#     shuffle=False
# )

# class_names = test_ds.class_names

# # -------------------------
# # Evaluate Accuracy
# # -------------------------
# loss, accuracy = model.evaluate(test_ds)
# print("Test Accuracy:", accuracy)

# # -------------------------
# # Predictions
# # -------------------------
# y_true = np.concatenate([y for x, y in test_ds], axis=0)
# y_pred_probs = model.predict(test_ds)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # -------------------------
# # Classification Report
# # -------------------------
# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))

# # -------------------------
# # Confusion Matrix
# # -------------------------
# cm = confusion_matrix(y_true, y_pred)

# plt.figure()
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # -------------------------
# # Latency Test
# # -------------------------
# sample_batch = next(iter(test_ds))[0]

# start = time.time()
# model.predict(sample_batch)
# end = time.time()

# latency = (end - start) / len(sample_batch)
# print("Average Inference Time per Image (seconds):", latency)










# # resnet
# import tensorflow as tf
# import numpy as np
# import os
# import time
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load model
# model = tf.keras.models.load_model("crop_disease_resnet50_model")

# IMG_SIZE = 224
# TEST_DIR = r"C:\projects\cropdiseasedetection\test_dataset"

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32,
#     shuffle=False
# )

# class_names = test_ds.class_names

# # -------------------------
# # Evaluate Accuracy
# # -------------------------
# loss, accuracy = model.evaluate(test_ds)
# print("Test Accuracy:", accuracy)

# # -------------------------
# # Predictions
# # -------------------------
# y_true = np.concatenate([y for x, y in test_ds], axis=0)
# y_pred_probs = model.predict(test_ds)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # -------------------------
# # Classification Report
# # -------------------------
# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))

# # -------------------------
# # Confusion Matrix
# # -------------------------
# cm = confusion_matrix(y_true, y_pred)

# plt.figure()
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # -------------------------
# # Latency Test
# # -------------------------
# sample_batch = next(iter(test_ds))[0]

# start = time.time()
# model.predict(sample_batch)
# end = time.time()

# latency = (end - start) / len(sample_batch)
# print("Average Inference Time per Image (seconds):", latency)


















#cnn
#  import tensorflow as tf
# import numpy as np
# import os
# import time
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load model
# model = tf.keras.models.load_model("crop_disease_cnn_model.keras")

# IMG_SIZE = 224
# TEST_DIR = r"C:\projects\cropdiseasedetection\test_dataset"

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32,
#     shuffle=False
# )

# class_names = test_ds.class_names

# # -------------------------
# # Evaluate Accuracy
# # -------------------------
# loss, accuracy = model.evaluate(test_ds)
# print("Test Accuracy:", accuracy)

# # -------------------------
# # Predictions
# # -------------------------
# y_true = np.concatenate([y for x, y in test_ds], axis=0)
# y_pred_probs = model.predict(test_ds)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # -------------------------
# # Classification Report
# # -------------------------
# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))

# # -------------------------
# # Confusion Matrix
# # -------------------------
# cm = confusion_matrix(y_true, y_pred)

# plt.figure()
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # -------------------------
# # Latency Test
# # -------------------------
# sample_batch = next(iter(test_ds))[0]

# start = time.time()
# model.predict(sample_batch)
# end = time.time()

# latency = (end - start) / len(sample_batch)
# print("Average Inference Time per Image (seconds):", latency)






