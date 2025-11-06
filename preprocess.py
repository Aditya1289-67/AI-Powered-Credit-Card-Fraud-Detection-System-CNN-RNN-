# src/preprocess.py
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Paths
RAW_PATH = Path(__file__).resolve().parents[1] / "data/raw_transactions.csv"
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data/processed/transactions_processed.csv"

# Features and label
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL = "Class"

def preprocess():
    # 1️⃣ Load CSV
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]  # remove extra spaces

    # 2️⃣ Sort by time
    if "Time" in df.columns:
        df = df.sort_values("Time").reset_index(drop=True)

    # 3️⃣ Fill missing values
    df[FEATURES] = df[FEATURES].fillna(0)
    df[LABEL] = df[LABEL].fillna(0)

    # 4️⃣ Normalize features (V1-V28 + Amount)
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    # 5️⃣ Save processed CSV
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed CSV saved to: {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess()
