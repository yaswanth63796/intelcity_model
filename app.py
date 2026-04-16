import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
# Enable CORS so the React frontend can call this API
CORS(app)

# Load model
MODEL_PATH = "civic_model.h5"
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"Failed to load model: {e}")

# Class labels
class_labels = [
    "Domestic_trash",
    "Infrastructure_Damage_Concrete",
    "Parking_Issues_Illegal_Parking",
    "Road_Issues_Damaged_Sign",
    "Road_Issues_Pothole",
    "Vandalism_Graffiti"
]

IMG_SIZE = 224

def map_label_to_admin_format(label):
    mapping = {
        "Domestic_trash": {
            "category": "Waste & Cleaning", 
            "status": "Garbage", 
            "description": "Domestic Trash Accumulated"
        },
        "Infrastructure_Damage_Concrete": {
            "category": "Road & Infrastructure", 
            "status": "Damaged", 
            "description": "Concrete Infrastructure Damage"
        },
        "Parking_Issues_Illegal_Parking": {
            "category": "Traffic & Parking", 
            "status": "Violation", 
            "description": "Illegal Parking Issue"
        },
        "Road_Issues_Damaged_Sign": {
            "category": "Street Signs & Safety", 
            "status": "Damaged", 
            "description": "Damaged Street Sign"
        },
        "Road_Issues_Pothole": {
            "category": "Road Maintenance", 
            "status": "Damaged", 
            "description": "Road Pothole"
        },
        "Vandalism_Graffiti": {
            "category": "Public Property", 
            "status": "Vandalized", 
            "description": "Vandalism/Graffiti"
        }
    }
    return mapping.get(label, {"category": "General", "status": "Unknown", "description": "Unknown Issue"})

def preprocess_image(image_bytes):
    # Convert bytes directly to Image using PIL to avoid saving files to disk (faster & cleaner)
    image_obj = Image.open(io.BytesIO(image_bytes))
    if image_obj.mode != "RGB":
        image_obj = image_obj.convert("RGB")
    
    image_obj = image_obj.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(image_obj)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalization
    return img_array

# Only keeping the API route, NO index.html routing
@app.route("/analyze", methods=["POST"])
def analyze_image():
    if model is None:
        return jsonify({"error": "AI Model is not loaded on the server"}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected for analysis"}), 400
        
    try:
        # Read the image bytes directly instead of saving to UPLOAD_FOLDER
        img_bytes = file.read()
        processed_img = preprocess_image(img_bytes)
        
        # Predict
        preds = model.predict(processed_img)[0]
        
        # Get highest confidence class
        class_index = np.argmax(preds)
        confidence = float(np.max(preds)) * 100
        
        # If the model is not confident, mark it as irrelevant/undamaged
        if confidence < 60.0:
            predicted_label = "Unrecognized"
            admin_info = {
                "category": "Unrecognized", 
                "status": "undamaged", 
                "description": "The model cannot predict the class"
            }
        else:
            predicted_label = class_labels[class_index]
            admin_info = map_label_to_admin_format(predicted_label)
        
        return jsonify({
            "success": True,
            "raw_label": predicted_label,
            "confidence": f"{confidence:.2f}",
            "context": admin_info['category'],
            "status": admin_info['status'],
            "problem": admin_info['description']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
