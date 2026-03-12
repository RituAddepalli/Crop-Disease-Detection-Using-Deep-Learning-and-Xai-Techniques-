# this  is to run on locally but now using predict_xai_web.py file 
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ------------------------------
# LOAD FIXED MODEL
# ------------------------------
# model = tf.keras.models.load_model("crop_disease_mobilenetv2_FIXED_FINAL.keras")
model = tf.keras.models.load_model("best_crop_disease_model.keras")

print("✔ Model loaded")
print("\nModel Layers:")
for l in model.layers:
    print(l.name)

# ------------------------------
# CLASS NAMES
# ------------------------------
dataset_dir = r"C:\projects\cropdiseasedetection\dataset"
class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
print("✔ Classes:", class_names)

# ------------------------------
# LAST CONV LAYER (CORRECT FOR YOUR MODEL)
# ------------------------------
last_conv_name = "block_16_project"
last_conv_layer = model.get_layer(last_conv_name)
print("✔ Last Conv Layer:", last_conv_name)

# ------------------------------
# GRAD-CAM FUNCTION
# ------------------------------
def generate_gradcam(image_path):

    # Load & preprocess
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Build grad model
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_out)[0]
    conv_out = conv_out[0]

    # Weighted sum
    weights = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)

    for i in range(weights.shape[0]):
        heatmap += weights[i] * conv_out[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-9)

    # Overlay
    original = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    result = cv2.addWeighted(original, 0.3, heatmap, 0.7, 0)

    return int(pred_index), result

# ------------------------------
# PREDICT FUNCTION
# ------------------------------
def predict(image_path):
    idx, grad_img = generate_gradcam(image_path)
    predicted_class = class_names[idx]

    print("\nPrediction:", predicted_class)

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(grad_img, cv2.COLOR_BGR2RGB))
    plt.title(predicted_class)
    plt.axis("off")
    plt.show()

# def predict(image_path):
#     idx, grad_img = generate_gradcam(image_path)
#     predicted_class = class_names[idx]

#     print("\nPrediction:", predicted_class)

#     plt.imshow(cv2.cvtColor(grad_img, cv2.COLOR_BGR2RGB))
#     plt.axis("off")
#     plt.show()
    

# ------------------------------
# RUN ON TEST FOLDER
# ------------------------------
test_folder = r"C:\projects\cropdiseasedetection\predict_dataset"

for file in os.listdir(test_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(test_folder, file)
        print("\n=== Testing:", path, "===")
        predict(path)

print("\n✔ Completed.")




















# import tensorflow as tf
# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt

# # ------------------------------
# # LOAD FIXED MODEL
# # ------------------------------
# # model = tf.keras.models.load_model("crop_disease_mobilenetv2_FIXED_FINAL.keras")
# model = tf.keras.models.load_model("best_crop_disease_model.keras")

# print("✔ Model loaded")

# # ------------------------------
# # CLASS NAMES
# # ------------------------------
# dataset_dir = r"C:\projects\cropdiseasedetection\dataset"
# class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
# print("✔ Classes:", class_names)

# # ------------------------------
# # LAST CONV LAYER (CORRECT FOR YOUR MODEL)
# # ------------------------------
# last_conv_name = "block_16_project"
# last_conv_layer = model.get_layer(last_conv_name)
# print("✔ Last Conv Layer:", last_conv_name)

# # ------------------------------
# # GRAD-CAM FUNCTION
# # ------------------------------
# def generate_gradcam(image_path):

#     # Load & preprocess
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)

#     # Build grad model
#     grad_model = tf.keras.Model(
#         inputs=model.input,
#         outputs=[last_conv_layer.output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_out, predictions = grad_model(img_array)
#         pred_index = tf.argmax(predictions[0])
#         loss = predictions[:, pred_index]

#     grads = tape.gradient(loss, conv_out)[0]
#     conv_out = conv_out[0]

#     # Weighted sum
#     weights = tf.reduce_mean(grads, axis=(0, 1))
#     heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)

#     for i in range(weights.shape[0]):
#         heatmap += weights[i] * conv_out[:, :, i]

#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= (heatmap.max() + 1e-9)

#     # Overlay
#     original = cv2.imread(image_path)
#     heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
#     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

#     result = cv2.addWeighted(original, 0.3, heatmap, 0.7, 0)

#     return int(pred_index), result

# # ------------------------------
# # PREDICT FUNCTION
# # ------------------------------
# def predict(image_path):
#     idx, grad_img = generate_gradcam(image_path)
#     predicted_class = class_names[idx]

#     print("\nPrediction:", predicted_class)

#     plt.imshow(cv2.cvtColor(grad_img, cv2.COLOR_BGR2RGB))
#     plt.axis("off")
#     plt.show()

# # ------------------------------
# # RUN ON TEST FOLDER
# # ------------------------------
# test_folder = r"C:\projects\cropdiseasedetection\test_dataset"

# for file in os.listdir(test_folder):
#     if file.lower().endswith((".jpg", ".jpeg", ".png")):
#         path = os.path.join(test_folder, file)
#         print("\n=== Testing:", path, "===")
#         predict(path)

# print("\n✔ Completed.")






# import tensorflow as tf
# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt

# # ------------------------------
# # LOAD MODEL
# # ------------------------------
# model = tf.keras.models.load_model("crop_disease_mobilenetv2_functional.keras")
# print("✔ Model loaded")

# # ------------------------------
# # CLASS NAMES
# # ------------------------------
# dataset_dir = r"C:\projects\cropdiseasedetection\dataset"
# class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
# print("✔ Classes:", class_names)

# # ------------------------------
# # LAST CONV LAYER (direct in main model)
# # ------------------------------
# last_conv_layer_name = "block_16_project"
# last_conv_layer = model.get_layer(last_conv_layer_name)
# print("✔ Last Conv Layer:", last_conv_layer_name)

# # ------------------------------
# # GRAD-CAM FUNCTION
# # ------------------------------
# def generate_gradcam(image_path):

#     # Load image
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)

#     # Grad-CAM model
#     grad_model = tf.keras.Model(
#         inputs=model.input,
#         outputs=[last_conv_layer.output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_output, predictions = grad_model(img_array)
#         pred_index = tf.argmax(predictions[0])
#         loss = predictions[:, pred_index]

#     grads = tape.gradient(loss, conv_output)

#     if grads is None:
#         raise ValueError("❌ Gradients None — but this version SHOULD NOT fail.")

#     grads = grads[0]
#     conv_output = conv_output[0]

#     weights = tf.reduce_mean(grads, axis=(0, 1))
#     heatmap = np.zeros(conv_output.shape[:2], dtype=np.float32)

#     for i in range(weights.shape[0]):
#         heatmap += weights[i] * conv_output[:, :, i]

#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= (heatmap.max() + 1e-9)

#     original = cv2.imread(image_path)
#     heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
#     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

#     final = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

#     return int(pred_index), final


# # ------------------------------
# # PREDICT FUNCTION
# # ------------------------------
# def predict(image_path):
#     idx, grad_img = generate_gradcam(image_path)
#     predicted_class = class_names[idx]

#     print("\nPrediction:", predicted_class)

#     plt.imshow(cv2.cvtColor(grad_img, cv2.COLOR_BGR2RGB))
#     plt.axis("off")
#     plt.show()


# # ------------------------------
# # RUN ON TEST FOLDER
# # ------------------------------
# test_folder = r"C:\projects\cropdiseasedetection\test_dataset"

# for file in os.listdir(test_folder):
#     if file.lower().endswith((".jpg", ".jpeg", ".png")):
#         path = os.path.join(test_folder, file)
#         print("\n=== Testing:", path, "===")
#         predict(path)

# print("\n✔ Completed.")
