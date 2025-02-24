import pandas as pd
import joblib
from datetime import datetime
import json

# Load the trained model and scaler
rf_model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")  # Ensure the scaler was saved

# Load new transactions for compliance check
# Ensure the new transactions file is in the same directory
new_transactions = pd.read_csv("new_transactions.csv")

# Convert transaction_time to datetime
new_transactions["transaction_time"] = pd.to_datetime(new_transactions["transaction_time"])

# Feature Engineering
new_transactions["hour"] = new_transactions["transaction_time"].dt.hour

# Select relevant features
features = ["amount", "hour"]
X_new = new_transactions[features]

# Check if X_new is empty
if X_new.empty:
    print("No new transactions to process.")
else:
    # Standardize numerical features
    X_new_scaled = scaler.transform(X_new)

    # Predict fraud
    new_transactions["is_fraud"] = rf_model.predict(X_new_scaled)

    # Flag suspicious transactions
    flagged_transactions = new_transactions[new_transactions["is_fraud"] == 1]

    # Convert timestamps to strings for JSON serialization
    flagged_transactions["transaction_time"] = flagged_transactions["transaction_time"].astype(str)

    # Generate a report summarizing flagged transactions and any actions taken
    report = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_transactions": len(new_transactions),
        "flagged_transactions": len(flagged_transactions),
        "flagged_details": flagged_transactions.to_dict(orient="records")
    }

    # Save report as JSON
    report_filename = "fraud_detection_report.json"
    with open(report_filename, "w") as report_file:
        json.dump(report, report_file, indent=4)

    print(f"Report generated and saved as {report_filename}")
