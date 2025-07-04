from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import os

import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import img_to_array  # type: ignore

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'best_model.h5'
CONFIDENCE_THRESHOLD = 0.95
IRRELEVANT_MESSAGE = "The uploaded image is likely not a relevant plant image."


CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

IMG_SIZE = (160, 160)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f" Failed to load model: {e}")
    model = None


def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).resize(IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f" Error during preprocessing: {e}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        print("Model not loaded.")
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        print("No image part in the request.")
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    print(f"📥 Received file: {image_file.filename}")

    if not image_file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed'}), 400

    try:
        image_bytes = image_file.read()
        print("Image file read successfully.")
    except Exception as e:
        return jsonify({'error': 'Error reading uploaded image'}), 400

    processed_image = preprocess_image(image_bytes)
    if processed_image is None:
        return jsonify({'error': 'Failed to process image'}), 400

    try:
        predictions = model.predict(processed_image)
        print(f"Predictions: {predictions}")

        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index])

        print(f"Predicted index: {predicted_class_index}, confidence: {confidence}")

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({'prediction': IRRELEVANT_MESSAGE})
        else:
            return jsonify({
                'prediction': CLASS_NAMES[predicted_class_index],
                'confidence': confidence
            })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/')
def home():
    return '🌱 KilimoScan is Up and Running — Made with Tiffany & your boy ByteCraft404!'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
