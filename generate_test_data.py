import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic test data
np.random.seed(42)
n_records = 500  # Use 500 records for testing

data = {
    "transaction_id": range(1, n_records + 1),
    "transaction_time": [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)],
    "amount": np.random.uniform(1, 1000, n_records),
    "id_fraud": np.random.choice([0, 1], n_records, p=[0.95, 0.05])
}

df = pd.DataFrame(data)
df.to_csv("test_fraud_dataset.csv", index=False)
print("Test data created and saved as test_fraud_dataset.csv.")
