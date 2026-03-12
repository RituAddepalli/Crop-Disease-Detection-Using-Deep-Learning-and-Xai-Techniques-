from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import sys
import time

# ✅ app.py is in backend/, go one level up to reach project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Frontend folder where index.html and style.css live
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# ✅ predict_xai_web.py is in src/
sys.path.append(os.path.join(ROOT_DIR, "src"))

from predict_xai_web import predict_for_web

# ✅ Tell Flask to serve static files from frontend/ folder
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

UPLOAD_FOLDER = os.path.join(ROOT_DIR, "backend", "temp")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Sample images folder
SAMPLE_DIR = os.path.join(ROOT_DIR, "dataset", "sample_web_images")


# ✅ Serve index.html at root URL
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ✅ Serve sample images from dataset/sample_web_images/
@app.route("/samples/<filename>")
def get_sample(filename):
    return send_from_directory(SAMPLE_DIR, filename)


# ✅ List all sample images so frontend can load them dynamically
@app.route("/samples")
def list_samples():
    files = sorted([
        f for f in os.listdir(SAMPLE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return jsonify({"samples": files})


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    predicted_class, grad_img = predict_for_web(path)

    # Timestamp makes filename unique → fixes browser cache problem
    timestamp = int(time.time())
    grad_filename = f"gradcam_{timestamp}.jpg"
    grad_path = os.path.join(UPLOAD_FOLDER, grad_filename)
    cv2.imwrite(grad_path, grad_img)

    return jsonify({
        "prediction": predicted_class,
        "gradcam": f"/temp/{grad_filename}"
    })


@app.route("/temp/<filename>")
def get_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import cv2
# import sys
# import time

# # ✅ app.py is in backend/, go one level up to reach project root
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # ✅ Frontend folder where index.html and style.css live
# FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# # ✅ Allow importing predict_xai_web.py from project root
# sys.path.append(os.path.join(ROOT_DIR, "src"))

# from predict_xai_web import predict_for_web

# # ✅ Tell Flask to serve static files from frontend/ folder
# app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
# CORS(app)

# UPLOAD_FOLDER = os.path.join(ROOT_DIR, "backend", "temp")
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# # ✅ Serve index.html at root URL
# @app.route("/")
# def home():
#     return send_from_directory(FRONTEND_DIR, "index.html")


# @app.route("/predict", methods=["POST"])
# def predict():

#     if "image" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["image"]

#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400

#     path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(path)

#     predicted_class, grad_img = predict_for_web(path)

#     # Timestamp makes filename unique → fixes browser cache problem
#     timestamp = int(time.time())
#     grad_filename = f"gradcam_{timestamp}.jpg"
#     grad_path = os.path.join(UPLOAD_FOLDER, grad_filename)
#     cv2.imwrite(grad_path, grad_img)

#     return jsonify({
#         "prediction": predicted_class,
#         "gradcam": f"/temp/{grad_filename}"
#     })


# @app.route("/temp/<filename>")
# def get_image(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)


# if __name__ == "__main__":
#     app.run(debug=True)