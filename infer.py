import torch
import pickle
import numpy as np
import pandas as pd
from .models.cnn_rnn_model import CNNRNNFraudDetector
from .utils import calculate_heuristic

class FraudDetector:
    def __init__(self, model_path, scaler_path, device=None, seq_len=20, batch_size=128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        self.batch_size = batch_size

        # Load model
        self.model = CNNRNNFraudDetector(seq_len=seq_len)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Load scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def _create_sequences_full(self, X):
        n = len(X)
        seqs = []
        for i in range(n):
            start_idx = max(0, i - self.seq_len + 1)
            seq = X[start_idx:i+1]
            if len(seq) < self.seq_len:
                pad = np.zeros((self.seq_len - len(seq), X.shape[1]), dtype=np.float32)
                seq = np.vstack((pad, seq))
            seqs.append(seq)
        return np.array(seqs, dtype=np.float32)

    def predict_chunk(self, df_chunk):
        feature_cols = [c for c in df_chunk.columns if c.startswith("V") or c == "Amount"]
        X = df_chunk[feature_cols].astype(np.float32).values
        X_scaled = self.scaler.transform(X)
        sequences = self._create_sequences_full(X_scaled)
        preds_out = []

        with torch.no_grad():
            for i in range(0, len(sequences), self.batch_size):
                batch_seq = sequences[i:i+self.batch_size]
                batch_tensor = torch.tensor(batch_seq, dtype=torch.float32).to(self.device)
                logits = self.model(batch_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds_out.extend([p[-1] for p in probs])

        return np.array(preds_out)

    def predict_file(self, df, chunksize=5000):
        results = []
        total_rows = len(df)
        for start in range(0, total_rows, chunksize):
            end = start + chunksize
            df_chunk = df.iloc[start:end]
            preds = self.predict_chunk(df_chunk)
            results.extend(preds)
        return np.array(results[:total_rows])

    def most_responsible_feature(self, fraud_row, nonfraud_rows):
        """
        Calculate the feature with highest deviation from mean of non-fraud rows (heuristic)
        """
        feature_cols = [c for c in fraud_row.index if c.startswith("V") or c=="Amount"]
        return calculate_heuristic(fraud_row[feature_cols], nonfraud_rows[feature_cols])
