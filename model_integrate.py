import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Data Preprocessing
# Load the dataset
data = pd.read_csv('synthetic_fraud_dataset.csv')

# Check the columns in the dataset
print(data.columns)

# Ensure the column names match those in the dataset
X = data[['transaction_time', 'amount']]
y = data['id_fraud']

# Convert transaction_time to datetime and extract useful features
X['transaction_time'] = pd.to_datetime(X['transaction_time'])
X['hour'] = X['transaction_time'].dt.hour
X['day_of_week'] = X['transaction_time'].dt.dayofweek
X['day_of_month'] = X['transaction_time'].dt.day

# Drop the original transaction_time column
X = X.drop(columns=['transaction_time'])

# Standardize the amount feature
scaler = StandardScaler()
X['amount'] = scaler.fit_transform(X[['amount']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Training
# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Step 3: Compliance Workflow Integration
def check_transactions(transactions):
    # Preprocess transactions
    transactions['transaction_time'] = pd.to_datetime(transactions['transaction_time'])
    transactions['hour'] = transactions['transaction_time'].dt.hour
    transactions['day_of_week'] = transactions['transaction_time'].dt.dayofweek
    transactions['day_of_month'] = transactions['transaction_time'].dt.day
    transactions = transactions.drop(columns=['transaction_time'])
    transactions['amount'] = scaler.transform(transactions[['amount']])
    
    # Predict fraud
    predictions = clf.predict(transactions)
    
    # Flag suspicious transactions
    flagged_transactions = transactions[predictions == 1]
    
    return flagged_transactions

# Generate a report summarizing flagged transactions and actions taken
def generate_report(flagged_transactions):
    report = f"Compliance Report - {datetime.datetime.now()}\n"
    report += "---------------------------------------------------\n"
    report += f"Total Flagged Transactions: {len(flagged_transactions)}\n"
    report += "\nFlagged Transactions Details:\n"
    report += flagged_transactions.to_string()
    
    with open('compliance_report.txt', 'w') as file:
        file.write(report)
    
    print("Compliance report generated and saved as 'compliance_report.txt'.")

# Example usage
# Load a new batch of transactions to check
new_transactions = pd.DataFrame({
    'transaction_time': ['2025-01-01 12:30:00', '2025-01-01 14:45:00'],
    'amount': [500, 1500000]
})

flagged_transactions = check_transactions(new_transactions)
generate_report(flagged_transactions)
