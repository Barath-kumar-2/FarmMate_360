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
    season = season.lower()

    if season == "kharif":
        return {
            "temperature": random.uniform(25, 35),
            "humidity": random.uniform(70, 90),
            "rainfall": random.uniform(150, 300)
        }
    elif season == "rabi":
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

# --- Helper: default soil values ---
def get_soil_values(pincode):
    # Future: replace with real soil API / dataset
    return {
        "N": random.randint(30, 70),
        "P": random.randint(20, 60),
        "K": random.randint(20, 60),
        "ph": round(random.uniform(5.5, 7.5), 2)
    }

# --- Reasoning engine ---
def get_reason(crop, weather, soil, farm_size):
    reasons = []

    # Weather-based reasoning
    if weather["rainfall"] > 150:
        reasons.append("Suitable for high rainfall")
    elif weather["rainfall"] < 50:
        reasons.append("Suitable for low rainfall conditions")

    # Soil-based reasoning
    if 6 <= soil["ph"] <= 7:
        reasons.append("Optimal soil pH")
    else:
        reasons.append("Tolerates varied soil pH")

    # Farm size reasoning
    if farm_size < 2:
        reasons.append("Good for small-scale farming")
    elif farm_size > 5:
        reasons.append("Suitable for large-scale farming")
    else:
        reasons.append("Moderate farm size compatible")

    return reasons

@app.route('/')
def home():
    return "FarmMate 360 Backend Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        pincode = data.get('pincode')
        season = data.get('season')
        farm_size = float(data.get('farm_size', 1))

        if not pincode or not season:
            return jsonify({"error": "Missing pincode or season"}), 400

        # Simulated inputs
        weather = get_weather(season)
        soil = get_soil_values(pincode)

        # Model input
        sample = pd.DataFrame([{
            'N': soil['N'],
            'P': soil['P'],
            'K': soil['K'],
            'temperature': weather['temperature'],
            'humidity': weather['humidity'],
            'ph': soil['ph'],
            'rainfall': weather['rainfall']
        }])

        # Get probabilities
        probs = model.predict_proba(sample)[0]
        classes = model.classes_

        # Top 5 crops
        top_indices = probs.argsort()[-5:][::-1]

        results = []
        for i in top_indices:
            crop = classes[i]
            prob = probs[i]

            results.append({
                "crop": crop,
                "confidence": round(float(prob) * 100, 2),
                "reasons": get_reason(crop, weather, soil, farm_size)
            })

        return jsonify({
            "pincode": pincode,
            "season": season,
            "farm_size": farm_size,
            "recommended_crops": results,
            "used_values": {
                "soil": soil,
                "weather": weather
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)