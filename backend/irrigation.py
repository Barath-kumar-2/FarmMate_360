import pandas as pd

# ================= LOAD MODEL =================
def load_irrigation_model(path):
    import pickle
    with open(path, 'rb') as f:
        pack = pickle.load(f)
    return pack["model"], pack["encoders"], pack["target"]

# ================= SOIL MOISTURE =================
def estimate_soil_moisture(rainfall, temp, humidity):
    moisture = (rainfall * 0.5) + (humidity * 0.3) - (temp * 0.2)
    return max(10, min(moisture, 90))

# ================= ML =================
def predict_level(model, encoders, target, data):
    df = pd.DataFrame([data])

    for col in ['crop_type', 'season']:
        df[col] = encoders[col].transform(df[col])

    pred = model.predict(df)[0]
    return target.inverse_transform([pred])[0]

# ================= PHYSICS =================
kc_map = {
    "rice": 1.1,
    "wheat": 0.9,
    "maize": 1.0,
    "cotton": 1.2,
    "sugarcane": 1.25
}

def calculate_water(crop, temp, humidity, rainfall, soil_moisture, area, flow):

    kc = kc_map.get(crop.lower(), 1.0)

    ET0 = 0.0023 * (temp + 17.8) * (temp - 10) * (1 - humidity/100)
    ETc = kc * ET0

    effective_rainfall = rainfall * 0.8

    if soil_moisture > 70:
        factor = 0.3
    elif soil_moisture > 40:
        factor = 0.6
    else:
        factor = 1.0

    water_mm = max(ETc - effective_rainfall, 0) * factor

    area_m2 = area * 4046.86
    water_litres = water_mm * area_m2

    time_hours = (water_litres / flow) / 3600

    return water_litres, time_hours

# ================= REASON =================
def generate_reason(temp, humidity, rainfall, soil_moisture, level):

    reasons = []

    if temp > 30:
        reasons.append("High temperature increases evaporation")

    if rainfall < 50:
        reasons.append("Low rainfall reduces water availability")

    if soil_moisture < 40:
        reasons.append("Soil moisture is low")

    if humidity < 40:
        reasons.append("Low humidity causes faster water loss")

    reasons.append(f"Irrigation classified as {level}")

    return reasons