from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import os

import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import img_to_array # type: ignore

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'best_model.h5'
CONFIDENCE_THRESHOLD = 0.95
IRRELEVANT_MESSAGE = "The uploaded image is likely not a relevant plant image."

try:
    model = load_model(MODEL_PATH)
    CLASS_NAMES = sorted(os.listdir("./Data/PlantVillage/train"))
    IMG_SIZE = (160, 160)
except Exception as e:
    model = None
    CLASS_NAMES = []
    IMG_SIZE = (160, 160)

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).resize(IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not CLASS_NAMES:
        print("Model not loaded or class names not found")
        return jsonify({'error': 'Model not loaded or class names not found'}), 500

    if 'image' not in request.files:
        print("No image part in the request")
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    print(f"📥 Received file: {image_file.filename}")

    if not image_file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        print("Invalid file type")
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are allowed'}), 400

    try:
        image_bytes = image_file.read()
        print("Image file read successfully")
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return jsonify({'error': 'Error reading uploaded image'}), 400

    processed_image = preprocess_image(image_bytes)
    if processed_image is None:
        print("Failed to preprocess image")
        return jsonify({'error': 'Failed to process image'}), 400

    try:
        predictions = model.predict(processed_image)
        print(f"Predictions: {predictions}")

        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])

        print(f"Predicted index: {predicted_class_index}, confidence: {confidence}")

        if confidence < CONFIDENCE_THRESHOLD:
            print("Low confidence - probably irrelevant image")
            return jsonify({'prediction': IRRELEVANT_MESSAGE}) 
        else:
            return jsonify({'prediction': CLASS_NAMES[predicted_class_index], 'confidence': confidence})

    except Exception as e:
        print(f"Error during model prediction: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
   
@app.route('/')
def home():
    return 'KilimoScan is Up and Running Made with Tiffany & by your boy ByteCraft404!'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)