import pickle
import pandas as pd

# Load model
with open('crop_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create input as DataFrame (with column names)
sample = pd.DataFrame([{
    'N': 90,
    'P': 40,
    'K': 40,
    'temperature': 25,
    'humidity': 80,
    'ph': 6.5,
    'rainfall': 200
}])

# Predict
prediction = model.predict(sample)

print("Recommended Crop:", prediction[0])