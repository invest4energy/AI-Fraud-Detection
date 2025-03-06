import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the test dataset
test_data = pd.read_csv('synthetic_fraud_test_dataset.csv')

# Separate features and target
X_test = test_data[['transaction_time', 'amount']]
y_test = test_data['id_fraud']

# Convert transaction_time to datetime and extract useful features
X_test['transaction_time'] = pd.to_datetime(X_test['transaction_time'])
X_test['hour'] = X_test['transaction_time'].dt.hour
X_test['day_of_week'] = X_test['transaction_time'].dt.dayofweek
X_test['day_of_month'] = X_test['transaction_time'].dt.day
X_test = X_test.drop(columns=['transaction_time'])

# Standardize the amount feature
scaler = StandardScaler()
X_test['amount'] = scaler.fit_transform(X_test[['amount']])

# Print the first few rows of X_test to verify
print(X_test.head())
