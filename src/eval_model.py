import tensorflow as tf
import numpy as np
import os
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# model is in saved_models/
model = tf.keras.models.load_model(
    os.path.join(ROOT_DIR, "saved_models", "best_crop_disease_model.keras")
)  # change the path to whatever saved model you want to evaluate
 
IMG_SIZE = 224

#  test dataset
TEST_DIR = os.path.join(ROOT_DIR, "dataset", "test_dataset")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    shuffle=False
)

class_names = test_ds.class_names

# Evaluate Accuracy

loss, accuracy = model.evaluate(test_ds)


# Predictions

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification Report

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))


# Overall Test Accuracy

print("\nOverall Test Accuracy:")
print(f"{accuracy:.4f}")


# Per-Crop Accuracy

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


# Confusion Matrix

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


# Latency Test

sample_batch = next(iter(test_ds))[0]

start = time.time()
model.predict(sample_batch)
end = time.time()

latency = (end - start) / len(sample_batch)

print("\nAverage Inference Time per Image (seconds):")
print(latency)

