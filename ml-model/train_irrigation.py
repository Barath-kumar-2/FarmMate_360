import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ================= LOAD DATA =================
file_path = os.path.join(os.path.dirname(__file__), '../data/irrigation_prediction.csv')

df = pd.read_csv(file_path)

print("✅ Data Loaded")
print(df.head())

# ================= CLEAN =================
df.columns = df.columns.str.strip().str.lower()

print("\nColumns:", df.columns.tolist())

# ================= RENAME =================
df.rename(columns={
    'soil_mois': 'soil_moisture',
    'temperat': 'temperature_c',
    'previous': 'previous_irrigation_mm'
}, inplace=True)

# ================= SELECT FEATURES =================
required_cols = [
    'soil_ph',
    'soil_moisture',
    'temperature_c',
    'humidity',
    'rainfall_mm',
    'crop_type',
    'season',
    'previous_irrigation_mm',
    'irrigation_need'
]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    print("❌ Missing columns:", missing)
    exit()

df = df[required_cols]

# Drop missing values
df.dropna(inplace=True)

print("\n✅ Cleaned Data:")
print(df.head())

# ================= ENCODING =================
encoders = {}

categorical_cols = ['crop_type', 'season']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df['irrigation_need'] = df['irrigation_need'].astype(str).str.strip().str.lower()
df['irrigation_need'] = target_encoder.fit_transform(df['irrigation_need'])

print("\n🎯 Target Classes:", target_encoder.classes_)

# ================= SPLIT =================
X = df.drop('irrigation_need', axis=1)
y = df['irrigation_need']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODEL =================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ================= EVALUATION =================
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\n📊 Train Accuracy: {train_acc:.2f}")
print(f"📊 Test Accuracy: {test_acc:.2f}")

# ================= SAVE =================
model_path = os.path.join(os.path.dirname(__file__), 'irrigation_model.pkl')

with open(model_path, 'wb') as f:
    pickle.dump({
        "model": model,
        "encoders": encoders,
        "target": target_encoder
    }, f)

print(f"\n💾 Model saved at: {model_path}")