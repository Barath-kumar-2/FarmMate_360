import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv('../data/Crop_recommendation.csv')

print("Dataset Loaded Successfully")
print(data.head())

# Features and target
X = data.drop('label', axis=1)
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")