import pandas as pd
import os

# ================= LOAD =================
file_path = os.path.join(os.path.dirname(__file__), '../data/rain_data.csv')

df = pd.read_csv(file_path)

print("Data Loaded:")
print(df.head())

# ================= CLEAN =================
df.columns = df.columns.str.strip().str.lower()

df['district'] = df['district'].str.strip()

# ================= SEASON MAP =================
SEASON_MAP = {
    "kharif": ["jun", "jul", "aug", "sep"],
    "rabi": ["oct", "nov", "dec", "jan"],
    "zaid": ["feb", "mar", "apr", "may"]
}

# ================= COMPUTE SEASONAL =================
rows = []

for _, row in df.iterrows():
    district = row['district']

    for season, months in SEASON_MAP.items():
        rainfall = sum([row[m] for m in months if m in row])

        rows.append({
            "district": district,
            "season": season,
            "monthly_rainfall": round(rainfall, 2)
        })

seasonal_df = pd.DataFrame(rows)

# ================= GROUP (IN CASE DUPLICATES) =================
seasonal_df = seasonal_df.groupby(['district','season'])['monthly_rainfall'].mean().reset_index()

print("\nFinal Output:")
print(seasonal_df.head())

# ================= SAVE =================
output_path = os.path.join(os.path.dirname(__file__), '../data/seasonal_rainfall.csv')

try:
    seasonal_df.to_csv(output_path, index=False)
except PermissionError:
    output_path = os.path.join(os.path.dirname(__file__), '../data/seasonal_rainfall_new.csv')
    seasonal_df.to_csv(output_path, index=False)

print("\n✅ Saved at:", output_path)