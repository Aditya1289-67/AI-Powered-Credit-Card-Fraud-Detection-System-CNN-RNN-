# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

from dataset import FraudSequenceDataset
from models.cnn_rnn_model import CNNRNNFraudDetector

# -----------------------------
# Config
# -----------------------------
SEQ_LEN = 20
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESH_RANGE = np.linspace(0.05, 0.5, 50)

# -----------------------------
# Dataset & DataLoader
# -----------------------------
dataset = FraudSequenceDataset(seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = MinMaxScaler()
all_features = dataset.df.iloc[:, 1:-1].values  # exclude Time and Class
dataset.df.iloc[:, 1:-1] = scaler.fit_transform(all_features)

# -----------------------------
# Model
# -----------------------------
model = CNNRNNFraudDetector(
    input_dim=29,
    seq_len=SEQ_LEN,
    rnn_hidden=128,
    cnn_channels=[32,64],
    rnn_layers=2,
    dropout=0.1
)
model.to(DEVICE)

# -----------------------------
# Loss & Optimizer
# -----------------------------
num_fraud = dataset.df['Class'].sum()
num_nonfraud = len(dataset.df) - num_fraud
pos_weight = torch.tensor([num_nonfraud / num_fraud]).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Helper: Best Threshold
# -----------------------------
def find_best_threshold(y_true, y_probs):
    best_thresh = 0.1
    best_f1 = 0
    for t in THRESH_RANGE:
        preds = (y_probs > t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh

# -----------------------------
# Training Loop
# -----------------------------
best_threshold = 0.1

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    all_labels, all_probs = [], []

    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    
    # tqdm for batch progress
    with tqdm(total=len(dataloader), desc="Training", ncols=100) as pbar:
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())

            # Update tqdm bar dynamically
            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
            pbar.update(1)

    # -----------------------------
    # Epoch Metrics
    # -----------------------------
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    best_threshold = find_best_threshold(all_labels, all_probs)
    preds = (all_probs > best_threshold).astype(int)

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    precision = precision_score(all_labels, preds)
    recall = recall_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    print(
        f"\n--- Epoch {epoch+1} Summary ---\n"
        f"Loss: {epoch_loss/len(dataloader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | "
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | AUC: {auc:.4f} | "
        f"Best Threshold: {best_threshold:.3f}"
    )

# -----------------------------
# Save Model
# -----------------------------
torch.save(model.state_dict(), "cnn_rnn_fraud_detector.pth")
print("\nModel saved to cnn_rnn_fraud_detector.pth")
