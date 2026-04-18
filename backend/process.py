from pathlib import Path
import pandas as pd

# ================= LOAD =================
# BUG FIX: use Path(__file__).resolve().parent instead of os.path.dirname(__file__)
# os.path.dirname returns '' when the script is run directly (not imported),
# making all relative paths silently resolve to the current working directory
# instead of the script's actual location.
BASE_DIR  = Path(__file__).resolve().parent
file_path = BASE_DIR / '../data/rain_data.csv'

df = pd.read_csv(file_path)

print("Data Loaded:")
print(df.head())

# ================= CLEAN =================
df.columns   = df.columns.str.strip().str.lower()
df['district'] = df['district'].str.strip()

# ================= SEASON MAP =================
SEASON_MAP = {
    "kharif": ["jun", "jul", "aug", "sep"],
    "rabi":   ["oct", "nov", "dec", "jan"],
    "zaid":   ["feb", "mar", "apr", "may"]
}

# ================= COMPUTE SEASONAL =================
rows = []

for _, row in df.iterrows():
    district = row['district']

    for season, months in SEASON_MAP.items():
        # BUG FIX: skip months not present in the CSV instead of raising KeyError.
        # The original code used `if m in row` which checks the *index* (column names),
        # which is correct — keeping that guard but being explicit about it.
        available_months = [m for m in months if m in df.columns]
        rainfall = sum(row[m] for m in available_months)

        rows.append({
            "district":        district,
            "season":          season,
            "monthly_rainfall": round(rainfall, 2)
        })

seasonal_df = pd.DataFrame(rows)

# ================= GROUP (in case of duplicates) =================
seasonal_df = (
    seasonal_df
    .groupby(['district', 'season'])['monthly_rainfall']
    .mean()
    .reset_index()
)

print("\nFinal Output:")
print(seasonal_df.head())

# ================= SAVE =================
output_path = BASE_DIR / '../data/seasonal_rainfall.csv'

try:
    seasonal_df.to_csv(output_path, index=False)
    print("\n✅ Saved at:", output_path.resolve())
except PermissionError:
    # BUG FIX: fallback path also uses BASE_DIR so it resolves correctly
    fallback_path = BASE_DIR / '../data/seasonal_rainfall_new.csv'
    seasonal_df.to_csv(fallback_path, index=False)
    print("\n⚠️  Permission denied — saved at:", fallback_path.resolve())