import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib  # For saving the trained model

# Load dataset
file_path = "synthetic_fraud_dataset.csv"  # Ensure the file is in the same directory
df = pd.read_csv(file_path)

# Convert transaction_time to datetime
df["transaction_time"] = pd.to_datetime(df["transaction_time"])

# Feature Engineering
df["hour"] = df["transaction_time"].dt.hour  # Extract hour as a feature

# Select relevant features and target
features = ["amount", "hour"]
X = df[features]
y = df["id_fraud"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(rf_model, "fraud_detection_model.pkl")
print("Model saved as fraud_detection_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
