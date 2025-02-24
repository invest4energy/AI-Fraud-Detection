import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model and scaler
rf_model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the test dataset
test_file_path = "test_fraud_dataset.csv"
test_df = pd.read_csv(test_file_path)

# Convert transaction_time to datetime
test_df["transaction_time"] = pd.to_datetime(test_df["transaction_time"])

# Feature Engineering
test_df["hour"] = test_df["transaction_time"].dt.hour

# Select relevant features and target
features = ["amount", "hour"]
X_test = test_df[features]
y_test = test_df["id_fraud"]

# Standardize numerical features
X_test_scaled = scaler.transform(X_test)

# Predict fraud
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
