# src/dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Paths
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data/processed/transactions_processed.csv"

# Features and label
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL = "Class"

class FraudSequenceDataset(Dataset):
    """
    Dataset for CNN→RNN fraud detection.
    Each sequence contains `seq_len` transactions.
    """

    def __init__(self, seq_len=20):
        self.seq_len = seq_len
        self.df = pd.read_csv(PROCESSED_PATH)

        # Sort by time if available
        if "Time" in self.df.columns:
            self.df = self.df.sort_values("Time").reset_index(drop=True)

        # Total number of sequences
        self.length = len(self.df) - seq_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Slice sequence
        seq_df = self.df.iloc[idx:idx+self.seq_len]

        # Features tensor: [seq_len, features]
        x = torch.tensor(seq_df[FEATURES].values, dtype=torch.float32)

        # Label: last transaction in the sequence
        y = torch.tensor(seq_df[LABEL].iloc[-1], dtype=torch.float32)

        return x, y
