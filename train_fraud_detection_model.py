import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib  # For saving the trained model
from sklearn.model_selection import GridSearchCV

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

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
rf = RandomForestClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_rf_model = grid_search.best_estimator_

# Train the best model
best_rf_model.fit(X_train, y_train)

# Predictions
y_pred_best = best_rf_model.predict(X_test)

# Evaluation
print("Best Model Classification Report:")
print(classification_report(y_test, y_pred_best))

# Save the best model
joblib.dump(best_rf_model, "best_fraud_detection_model.pkl")
print("Best model saved as best_fraud_detection_model.pkl")
