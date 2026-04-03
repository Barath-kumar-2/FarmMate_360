from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import random

app = Flask(__name__)
CORS(app)

# Load model
model_path = os.path.join(os.path.dirname(__file__), '../ml-model/crop_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# --- Helper: simulate weather based on season ---
def get_weather(season):
    if season.lower() == "kharif":
        return {
            "temperature": random.uniform(25, 35),
            "humidity": random.uniform(70, 90),
            "rainfall": random.uniform(150, 300)
        }
    elif season.lower() == "rabi":
        return {
            "temperature": random.uniform(15, 25),
            "humidity": random.uniform(40, 60),
            "rainfall": random.uniform(20, 100)
        }
    else:  # zaid
        return {
            "temperature": random.uniform(30, 40),
            "humidity": random.uniform(20, 50),
            "rainfall": random.uniform(10, 50)
        }

# --- Helper: default soil values (approximation) ---
def get_soil_values():
    return {
        "N": 50,
        "P": 40,
        "K": 40,
        "ph": 6.5
    }

@app.route('/')
def home():
    return "FarmMate 360 Backend Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    pincode = data.get('pincode')
    season = data.get('season')

    if not pincode or not season:
        return jsonify({"error": "Missing pincode or season"}), 400

    # Simulate weather
    weather = get_weather(season)

    # Get soil values
    soil = get_soil_values()

    # Combine inputs
    sample = pd.DataFrame([{
        'N': soil['N'],
        'P': soil['P'],
        'K': soil['K'],
        'temperature': weather['temperature'],
        'humidity': weather['humidity'],
        'ph': soil['ph'],
        'rainfall': weather['rainfall']
    }])

    prediction = model.predict(sample)

    return jsonify({
        "pincode": pincode,
        "season": season,
        "recommended_crop": prediction[0],
        "used_values": {
            "soil": soil,
            "weather": weather
        }
    })

if __name__ == '__main__':
    app.run(debug=True)