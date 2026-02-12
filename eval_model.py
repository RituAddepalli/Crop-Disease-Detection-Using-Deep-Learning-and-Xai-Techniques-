import tensorflow as tf
import numpy as np
import os
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = tf.keras.models.load_model("crop_disease_mobilenetv2_FIXED_FINAL.keras")

IMG_SIZE = 224
TEST_DIR = r"C:\projects\cropdiseasedetection\test_dataset"

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
print("Test Accuracy:", accuracy)

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
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------
# Latency Test
# -------------------------
sample_batch = next(iter(test_ds))[0]

start = time.time()
model.predict(sample_batch)
end = time.time()

latency = (end - start) / len(sample_batch)
print("Average Inference Time per Image (seconds):", latency)
