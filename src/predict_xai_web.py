


import tensorflow as tf
import numpy as np
import cv2
import os

# ✅ src/ is inside project root, so go one level up
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------
# LOAD MODEL
# ------------------------------
model = tf.keras.models.load_model(
    os.path.join(ROOT_DIR, "saved_models", "best_crop_disease_model.keras")
)
print("✔ Model loaded")

# ------------------------------
# CLASS NAMES
# ✅ dataset moved to dataset/train_dataset/
# ------------------------------
dataset_dir = os.path.join(ROOT_DIR, "dataset", "train_dataset")

class_names = sorted([
    d for d in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, d))
])

print("✔ Classes:", class_names)

# ------------------------------
# LAST CONV LAYER
# ------------------------------
last_conv_name = "block_16_project"
last_conv_layer = model.get_layer(last_conv_name)

# ------------------------------
# GRAD CAM
# ------------------------------
def generate_gradcam(image_path):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

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

    weights = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)

    for i in range(weights.shape[0]):
        heatmap += weights[i] * conv_out[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-9)

    original = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    result = cv2.addWeighted(original, 0.3, heatmap, 0.7, 0)

    return int(pred_index), result


# ------------------------------
# PREDICT FUNCTION
# ------------------------------
def predict_for_web(image_path):
    idx, grad_img = generate_gradcam(image_path)
    predicted_class = class_names[idx]
    return predicted_class, grad_img


# import tensorflow as tf
# import numpy as np
# import cv2
# import os

# # ------------------------------
# # LOAD MODEL
# # ------------------------------
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# model = tf.keras.models.load_model(os.path.join(ROOT_DIR, "saved_models", "best_crop_disease_model.keras"))
# dataset_dir = os.path.join(ROOT_DIR, "dataset", "train_dataset")

# class_names = sorted([
#     d for d in os.listdir(dataset_dir)
#     if os.path.isdir(os.path.join(dataset_dir, d))
# ])

# # ------------------------------
# # LAST CONV LAYER
# # ------------------------------
# last_conv_name = "block_16_project"
# last_conv_layer = model.get_layer(last_conv_name)

# # ------------------------------
# # GRAD CAM
# # ------------------------------
# def generate_gradcam(image_path):

#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224,224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)

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

#     weights = tf.reduce_mean(grads, axis=(0,1))

#     heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)

#     for i in range(weights.shape[0]):
#         heatmap += weights[i] * conv_out[:,:,i]

#     heatmap = np.maximum(heatmap,0)
#     heatmap /= (heatmap.max()+1e-9)

#     original = cv2.imread(image_path)

#     heatmap = cv2.resize(heatmap,(original.shape[1],original.shape[0]))

#     heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

#     result = cv2.addWeighted(original,0.3,heatmap,0.7,0)

#     return int(pred_index), result


# # ------------------------------
# # PREDICT FUNCTION
# # ------------------------------
# def predict_for_web(image_path):

#     idx, grad_img = generate_gradcam(image_path)

#     predicted_class = class_names[idx]

#     return predicted_class, grad_img