from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import os
import requests

from irrigation import (
    load_irrigation_model,
    estimate_soil_moisture,
    predict_level,
    generate_reason
)

# ================= CONFIG =================
API_KEY = "ae6d4369621e4f36c60da4059267cb71"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, '../frontend'))
CORS(app)

# ================= LOAD MODELS =================
with open(os.path.join(BASE_DIR, '../ml-model/crop_model.pkl'), 'rb') as f:
    crop_model = pickle.load(f)

irrigation_model, irrigation_encoders, irrigation_target = load_irrigation_model(
    os.path.join(BASE_DIR, '../ml-model/irrigation_model.pkl')
)

# ================= LOAD DATA =================
pincode_df = pd.read_csv(os.path.join(BASE_DIR, '../data/pincode-dataset.csv'))
soil_df    = pd.read_csv(os.path.join(BASE_DIR, '../data/soil_data.csv'))
rain_df    = pd.read_csv(os.path.join(BASE_DIR, '../data/seasonal_rainfall.csv'))

pincode_df.columns = pincode_df.columns.str.lower().str.strip()
soil_df.columns    = soil_df.columns.str.lower().str.strip()
rain_df.columns    = rain_df.columns.str.lower().str.strip()

# ================= CLEAN SOIL =================
soil_df.rename(columns={
    'a district':        'district',
    'nitrogen value':    'n',
    'phosphorous value': 'p',
    'potassium value':   'k',
    'ph':                'ph'
}, inplace=True)

soil_df[['n', 'p', 'k', 'ph']] = soil_df[['n', 'p', 'k', 'ph']].apply(
    pd.to_numeric, errors='coerce'
)

soil_grouped = soil_df.groupby('district').mean(numeric_only=True).reset_index()

# ================= UTIL FUNCTIONS =================
def get_district(pincode):
    try:
        pincode = int(pincode)
    except (ValueError, TypeError):
        return None

    pin_col  = next((c for c in ['pincode', 'pin'] if c in pincode_df.columns), None)
    dist_col = next((c for c in ['district', 'districtname'] if c in pincode_df.columns), None)

    if not pin_col or not dist_col:
        return None

    row = pincode_df[pincode_df[pin_col] == pincode]
    if row.empty:
        return None

    return str(row.iloc[0][dist_col]).strip()


def get_soil_values(pincode):
    district = get_district(pincode)
    if not district:
        return None

    row = soil_grouped[soil_grouped['district'].str.lower() == district.lower()]
    if row.empty:
        return None

    row = row.iloc[0]
    return {
        "N":        round(row['n'],  2),
        "P":        round(row['p'],  2),
        "K":        round(row['k'],  2),
        "ph":       round(row['ph'], 2) if pd.notna(row['ph']) else 7.0,
        "district": district
    }


def get_weather(district):
    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={district},IN&appid={API_KEY}&units=metric"
        )
        res  = requests.get(url, timeout=5)
        data = res.json()

        if res.status_code != 200:
            return {"temperature": 30, "humidity": 60}

        return {
            "temperature": data['main']['temp'],
            "humidity":    data['main']['humidity']
        }
    except Exception:
        return {"temperature": 30, "humidity": 60}


def get_rainfall(district, season):
    row = rain_df[
        (rain_df['district'].str.lower() == district.lower()) &
        (rain_df['season'].str.lower()   == season.lower())
    ]
    if row.empty:
        return 50.0
    return float(row.iloc[0]['monthly_rainfall'])


# ================= WATER CALCULATION =================
kc_map = {
    "rice":      1.10,
    "wheat":     0.90,
    "maize":     1.00,
    "cotton":    1.20,
    "sugarcane": 1.25
}

def calculate_water(crop, temp, humidity, rainfall, soil_moisture, area, flow):
    kc = kc_map.get(crop.lower(), 1.0)

    Rs  = 18 * (1 - 0.5 * (humidity / 100))
    ET0 = 0.0135 * (temp + 17.8) * Rs

    ETc = kc * ET0

    rainfall_daily = rainfall / 30
    effective_rain = rainfall_daily * 0.8

    if soil_moisture > 85:
        factor = 0.1
    elif soil_moisture > 70:
        factor = 0.3
    elif soil_moisture > 50:
        factor = 0.6
    else:
        factor = 1.0

    net_mm = max(ETc - effective_rain, 0) * factor * 4

    area_m2      = area * 4046.86
    water_litres = net_mm * area_m2

    if flow > 0 and water_litres > 0:
        time_hours = (water_litres / flow) / 3600
    else:
        time_hours = 0.0

    return water_litres, time_hours


# ================= ROUTES =================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop')
def crop_page():
    return render_template('crop.html')

@app.route('/irrigation')
def irrigation_page():
    return render_template('irrigation.html')

@app.route('/result')
def result_page():
    return render_template('result.html')


# ================= CROP RECOMMENDATION =================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        pincode   = data.get('pincode')
        season    = data.get('season')
        farm_size = float(data.get('farm_size', 1))

        if not pincode or not season:
            return jsonify({"error": "Missing required fields: pincode, season"}), 400

        soil = get_soil_values(pincode)
        if not soil:
            return jsonify({"error": "Invalid pincode or no soil data found"}), 400

        weather  = get_weather(soil["district"])
        rainfall = get_rainfall(soil["district"], season)

        sample = pd.DataFrame([{
            'N':           soil['N'],
            'P':           soil['P'],
            'K':           soil['K'],
            'temperature': weather['temperature'],
            'humidity':    weather['humidity'],
            'ph':          soil['ph'],
            'rainfall':    rainfall
        }])

        probs   = crop_model.predict_proba(sample)[0]
        classes = crop_model.classes_

        top_indices = probs.argsort()[-5:][::-1]

        results = []
        for i in top_indices:
            results.append({
                "crop":       classes[i],
                "confidence": round(float(probs[i]) * 100, 2),
                "reasons": [
                    "Based on soil nutrients",
                    "Weather conditions are suitable",
                    "Season compatibility"
                ]
            })

        return jsonify({
            "pincode":           pincode,
            "district":          soil["district"],
            "season":            season,
            "farm_size":         farm_size,
            "recommended_crops": results,
            "used_values": {
                "soil":    soil,
                "weather": {
                    "temperature": weather["temperature"],
                    "humidity":    weather["humidity"],
                    "rainfall":    rainfall
                }
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= IRRIGATION =================
@app.route('/predict_irrigation', methods=['POST'])
def predict_irrigation():
    try:
        data = request.json

        required = ['pincode', 'crop', 'season', 'area', 'flow']
        missing  = [f for f in required if not data.get(f) and data.get(f) != 0]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        pincode = data['pincode']
        crop    = data['crop']
        season  = data['season']
        area    = float(data['area'])
        flow    = float(data['flow'])

        soil = get_soil_values(pincode)
        if not soil:
            return jsonify({"error": "Invalid pincode or no soil data found"}), 400

        district = soil["district"]
        weather  = get_weather(district)
        rainfall = get_rainfall(district, season)

        soil_moisture = estimate_soil_moisture(
            rainfall,
            weather["temperature"],
            weather["humidity"]
        )

        try:
            level = predict_level(
                irrigation_model,
                irrigation_encoders,
                irrigation_target,
                {
                    "soil_ph":                soil["ph"],
                    "soil_moisture":          soil_moisture,
                    "temperature_c":          weather["temperature"],
                    "humidity":               weather["humidity"],
                    "rainfall_mm":            rainfall,
                    "crop_type":              crop.lower(),
                    "season":                 season.lower(),
                    "previous_irrigation_mm": rainfall * 0.3
                }
            )
        except ValueError as ve:
            return jsonify({"error": f"Irrigation model error: {ve}"}), 422

        water, time = calculate_water(
            crop,
            weather["temperature"],
            weather["humidity"],
            rainfall,
            soil_moisture,
            area,
            flow
        )

        water_per_acre = water / area if area > 0 else 0

        if soil_moisture < 35 or water_per_acre > 30000:
            level = "high"
        elif soil_moisture < 60 or water_per_acre > 15000:
            level = "medium"
        else:
            level = "low"

        if soil_moisture > 80:
            decision = "Skip irrigation today"
        elif soil_moisture > 60:
            decision = "Light irrigation recommended"
        else:
            decision = "Irrigation required"

        reasons = generate_reason(
            weather["temperature"],
            weather["humidity"],
            rainfall,
            soil_moisture,
            level
        )

        return jsonify({
            "water_litres":  round(water, 2),
            "time_hours":    round(time,  2),
            "level":         level,
            "decision":      decision,
            "soil_moisture": round(soil_moisture, 2),
            "ph":            soil["ph"],
            "reasons":       reasons
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= RUN =================
if __name__ == '__main__':
    app.run()