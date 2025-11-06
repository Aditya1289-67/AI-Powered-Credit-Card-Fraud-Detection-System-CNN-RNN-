# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from pathlib import Path

# Paths
RAW_PATH = Path(__file__).resolve().parents[1] / "data/raw_transactions.csv"
SCALER_PATH = Path(__file__).resolve().parents[1] / "data/scaler.pkl"

# Load raw data
df = pd.read_csv(RAW_PATH)

# Keep only numeric features: V1-V28 + Amount
features = df.iloc[:, 1:-1].values  # exclude Time & Class

# Scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Save scaler for inference
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"Scaler saved to: {SCALER_PATH}")
