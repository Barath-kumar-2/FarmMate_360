from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import requests

# 🔥 IMPORT IRRIGATION LOGIC
from irrigation import (
    load_irrigation_model,
    estimate_soil_moisture,
    predict_level,
    calculate_water,
    generate_reason
)

app = Flask(__name__)
CORS(app)

# ================= CONFIG =================
API_KEY = "ae6d4369621e4f36c60da4059267cb71"   # replace later

# ================= LOAD CROP MODEL =================
model_path = os.path.join(os.path.dirname(__file__), '../ml-model/crop_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# ================= LOAD IRRIGATION MODEL =================
irrigation_model_path = os.path.join(os.path.dirname(__file__), '../ml-model/irrigation_model.pkl')
irrigation_model, irrigation_encoders, irrigation_target = load_irrigation_model(irrigation_model_path)

# ================= LOAD DATA =================
pincode_df = pd.read_csv('../data/pincode-dataset.csv')
soil_df = pd.read_csv('../data/soil_data.csv')
rain_df = pd.read_csv('../data/seasonal_rainfall.csv')

# Normalize
pincode_df.columns = pincode_df.columns.str.lower().str.strip()
soil_df.columns = soil_df.columns.str.lower().str.strip()
rain_df.columns = rain_df.columns.str.lower().str.strip()

# ================= FIX SOIL =================
soil_df.rename(columns={
    'a district': 'district',
    'nitrogen value': 'n',
    'phosphorous value': 'p',
    'potassium value': 'k',
    'ph': 'ph'
}, inplace=True)

soil_df[['n','p','k','ph']] = soil_df[['n','p','k','ph']].apply(pd.to_numeric, errors='coerce')

soil_grouped = soil_df.groupby('district').mean(numeric_only=True).reset_index()

# ================= PINCODE → DISTRICT =================
def get_district(pincode):
    try:
        pincode = int(pincode)
    except:
        return None

    pin_col = next((c for c in ['pincode','pin','postalcode'] if c in pincode_df.columns), None)
    dist_col = next((c for c in ['district','districtname'] if c in pincode_df.columns), None)

    if not pin_col or not dist_col:
        return None

    row = pincode_df[pincode_df[pin_col] == pincode]
    if row.empty:
        return None

    return str(row.iloc[0][dist_col]).strip()

# ================= SOIL =================
def get_soil_values(pincode):
    district = get_district(pincode)
    if not district:
        return None

    row = soil_grouped[soil_grouped['district'].str.lower() == district.lower()]
    if row.empty:
        return None

    row = row.iloc[0]

    return {
        "N": round(row['n'], 2),
        "P": round(row['p'], 2),
        "K": round(row['k'], 2),
        "ph": round(row['ph'], 2),
        "district": district
    }

# ================= WEATHER =================
def get_weather(district):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={district},IN&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            return {"temperature": 30, "humidity": 60}

        return {
            "temperature": data['main']['temp'],
            "humidity": data['main']['humidity']
        }

    except:
        return {"temperature": 30, "humidity": 60}

# ================= RAINFALL =================
def get_rainfall(district, season):
    row = rain_df[
        (rain_df['district'].str.lower() == district.lower()) &
        (rain_df['season'].str.lower() == season.lower())
    ]

    if row.empty:
        return 50

    return float(row.iloc[0]['monthly_rainfall'])

# ================= ROUTES =================
@app.route('/')
def home():
    return "FarmMate 360 Backend Running"

# ================= 🌱 CROP =================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        pincode = data.get('pincode')
        season = data.get('season')
        farm_size = float(data.get('farm_size', 1))

        if not pincode or not season:
            return jsonify({"error": "Missing input"}), 400

        soil = get_soil_values(pincode)
        if not soil:
            return jsonify({"error": "Invalid pincode"}), 400

        weather = get_weather(soil["district"])
        rainfall = get_rainfall(soil["district"], season)

        sample = pd.DataFrame([{
            'N': soil['N'],
            'P': soil['P'],
            'K': soil['K'],
            'temperature': weather['temperature'],
            'humidity': weather['humidity'],
            'ph': soil['ph'],
            'rainfall': rainfall
        }])

        probs = model.predict_proba(sample)[0]
        classes = model.classes_

        top_indices = probs.argsort()[-5:][::-1]

        results = []
        for i in top_indices:
            results.append({
                "crop": classes[i],
                "confidence": round(float(probs[i]) * 100, 2),
                "reasons": [
                    "Based on soil nutrients",
                    "Weather conditions are suitable",
                    "Season compatibility"
                ]
            })

        return jsonify({
            "pincode": pincode,
            "district": soil["district"],
            "season": season,
            "farm_size": farm_size,
            "recommended_crops": results,
            "used_values": {
                "soil": soil,
                "weather": {
                    "temperature": weather["temperature"],
                    "humidity": weather["humidity"],
                    "rainfall": rainfall
                }
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= 💧 IRRIGATION =================
@app.route('/predict_irrigation', methods=['POST'])
def predict_irrigation():
    try:
        data = request.json

        pincode = data['pincode']
        crop = data['crop']
        season = data['season']
        area = float(data['area'])
        flow = float(data['flow'])

        soil = get_soil_values(pincode)
        if not soil:
            return jsonify({"error": "Invalid pincode"}), 400

        district = soil["district"]

        weather = get_weather(district)
        rainfall = get_rainfall(district, season)

        soil_moisture = estimate_soil_moisture(
            rainfall,
            weather["temperature"],
            weather["humidity"]
        )

        model_input = {
            "soil_ph": soil["ph"],
            "soil_moisture": soil_moisture,
            "temperature_c": weather["temperature"],
            "humidity": weather["humidity"],
            "rainfall_mm": rainfall,
            "crop_type": crop.lower(),
            "season": season.lower(),
            "previous_irrigation_mm": rainfall * 0.3
        }

        level = predict_level(
            irrigation_model,
            irrigation_encoders,
            irrigation_target,
            model_input
        )

        water, time = calculate_water(
            crop,
            weather["temperature"],
            weather["humidity"],
            rainfall,
            soil_moisture,
            area,
            flow
        )

        # 🔥 FIX: consistency
        if water < 1:
            water = 0
            time = 0
            level = "Low"

        reasons = generate_reason(
            weather["temperature"],
            weather["humidity"],
            rainfall,
            soil_moisture,
            level
        )

        return jsonify({
            "water_litres": round(water, 2),
            "time_hours": round(time, 2),
            "level": level,
            "soil_moisture": round(soil_moisture, 2),
            "ph": soil["ph"],
            "reasons": reasons
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= RUN =================
if __name__ == '__main__':
    print("System Ready 🚀")
    app.run(debug=True)