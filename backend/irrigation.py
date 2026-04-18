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
    """
    Calculate irrigation water requirement.

    Parameters
    ----------
    crop          : str   — crop name (matched against kc_map; defaults to 1.0)
    temp          : float — temperature in °C
    humidity      : float — relative humidity 0–100
    rainfall      : float — monthly rainfall in mm
    soil_moisture : float — estimated soil moisture 10–90
    area          : float — farm area in acres
    flow          : float — pump flow rate
    flow_unit     : str   — "lps" (litres/sec, default) or "lph" (litres/hour)

    Returns
    -------
    (water_litres, time_hours)

    Method: Hargreaves-simplified ET0, scaled by crop coefficient (Kc),
    adjusted for effective rainfall and current soil moisture.
    """

    kc = kc_map.get(crop.lower(), 1.0)

    # Solar radiation proxy: clear-sky Rs ~18 MJ/m²/day; cloud cover
    # (estimated from humidity) reduces it.
    Rs           = 18 * (1 - 0.5 * (humidity / 100))   # MJ/m²/day
    ET0_daily    = 0.0135 * (temp + 17.8) * Rs          # mm/day (Hargreaves)
    ET0_monthly  = ET0_daily * 30                        # mm/month

    ETc = kc * ET0_monthly                               # crop water need, mm/month

    effective_rainfall = rainfall * 0.8                  # 80 % effectiveness

    # Soil moisture adjustment factor
    if soil_moisture > 70:
        factor = 0.3
    elif soil_moisture > 40:
        factor = 0.6
    else:
        factor = 1.0

    net_irrigation_mm = max(ETc - effective_rainfall, 0) * factor

    area_m2       = area * 4046.86                       # acres → m²
    water_litres  = net_irrigation_mm * area_m2          # mm × m² = litres

    # BUG FIX: support both litres/sec and litres/hour so the caller can
    # pass whatever unit the pump spec uses without silent unit errors.
    if flow > 0 and water_litres > 0:
        if flow_unit == "lph":
            time_hours = water_litres / flow             # already in hours
        else:                                            # default: lps
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