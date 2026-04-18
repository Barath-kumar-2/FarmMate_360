import pandas as pd

# ================= LOAD MODEL =================
def load_irrigation_model(path):
    import pickle
    with open(path, 'rb') as f:
        pack = pickle.load(f)
    return pack["model"], pack["encoders"], pack["target"]

# ================= SOIL MOISTURE =================
def estimate_soil_moisture(rainfall, temp, humidity):
    rainfall_daily = rainfall / 30
    moisture = (rainfall_daily * 2) + (humidity * 0.3) - (temp * 0.2)
    return max(10, min(moisture, 90))

# ================= ML =================
def predict_level(model, encoders, target, data):
    df = pd.DataFrame([data])

    # BUG FIX: wrap encoder transforms in try/except so an unseen label
    # (e.g. a new crop variety) raises a clear ValueError instead of
    # an opaque sklearn exception that crashes the whole request.
    for col in ['crop_type', 'season']:
        try:
            df[col] = encoders[col].transform(df[col])
        except ValueError:
            known = list(encoders[col].classes_)
            raise ValueError(
                f"Unknown value '{df[col].iloc[0]}' for '{col}'. "
                f"Known values: {known}"
            )

    pred = model.predict(df)[0]
    return target.inverse_transform([pred])[0]

# ================= PHYSICS =================
kc_map = {
    "rice":      1.10,
    "wheat":     0.90,
    "maize":     1.00,
    "cotton":    1.20,
    "sugarcane": 1.25
}

def calculate_water(crop, temp, humidity, rainfall, soil_moisture, area, flow,
                    flow_unit="lps"):

    kc = kc_map.get(crop.lower(), 1.0)

    # ---- STEP 1: Solar Radiation ----
    Rs = 18 * (1 - 0.5 * (humidity / 100))   # MJ/m²/day

    # ---- STEP 2: ET0 (DAILY) ----
    ET0_daily = 0.0135 * (temp + 17.8) * Rs   # mm/day

    # ---- STEP 3: Crop ET (DAILY) ----
    ETc = kc * ET0_daily                      # mm/day

    # ---- STEP 4: Convert rainfall → DAILY ----
    rainfall_daily = rainfall / 30            # mm/day
    effective_rainfall = rainfall_daily * 0.8

    # ---- STEP 5: Soil Moisture Factor (IMPROVED) ----
    if soil_moisture > 85:
        factor = 0.1
    elif soil_moisture > 70:
        factor = 0.3
    elif soil_moisture > 50:
        factor = 0.6
    else:
        factor = 1.0

    # ---- STEP 6: Net Irrigation ----
    net_irrigation_mm = max(ETc - effective_rainfall, 0) * factor

    # ---- STEP 7: Area Conversion ----
    area_m2 = area * 4046.86

    # ---- STEP 8: Final Water (LITRES PER DAY) ----
    water_litres = net_irrigation_mm * area_m2

    # ---- STEP 9: Time Calculation ----
    if flow > 0 and water_litres > 0:
        if flow_unit == "lph":
            time_hours = water_litres / flow
        else:
            time_hours = (water_litres / flow) / 3600
    else:
        time_hours = 0.0

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

    # BUG FIX: always append the level so the list is never empty
    reasons.append(f"Irrigation classified as {level}")

    return reasons