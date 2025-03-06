import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Function to generate random datetime
def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

# Generate synthetic data
np.random.seed(42)
num_records = 1000
transaction_ids = range(1, num_records + 1)
transaction_times = [random_date(datetime(2023, 1, 1), datetime(2023, 12, 31)) for _ in range(num_records)]
transaction_amounts = np.random.uniform(1.0, 1000.0, num_records).round(2)
is_fraudulent = np.random.choice([0, 1], num_records, p=[0.95, 0.05])

# Create DataFrame
data = {
    'transaction_id': transaction_ids,
    'transaction_time': transaction_times,
    'transaction_amount': transaction_amounts,
    'is_fraudulent': is_fraudulent
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('fraud_detection_dataset.csv', index=False)
print("Synthetic dataset generated and saved as 'fraud_detection_dataset.csv'")
