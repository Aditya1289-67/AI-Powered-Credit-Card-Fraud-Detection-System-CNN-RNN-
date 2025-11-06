# src/models/cnn_rnn_model.py
import torch
import torch.nn as nn

class CNNRNNFraudDetector(nn.Module):
    """
    CNN → RNN Fraud Detection Model
    - CNN: extracts features per transaction
    - RNN: models sequences of transactions
    """

    def __init__(self, input_dim=29, cnn_channels=[32, 64], rnn_hidden=128, seq_len=20, rnn_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len

        # CNN: feature extraction per transaction
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU()
        )

        # RNN: sequence modeling (2 layers + dropout)
        self.rnn = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )

        # Fully connected output for each transaction
        self.fc = nn.Linear(rnn_hidden, 1)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        returns: logits [batch_size, seq_len] for BCEWithLogitsLoss
        """
        batch_size, seq_len, feat_dim = x.size()

        # CNN per transaction
        x = x.view(batch_size*seq_len, 1, feat_dim)       # [batch*seq, 1, features]
        cnn_out = self.cnn(x)                             # [batch*seq, cnn_channels[-1], features]
        cnn_out = torch.mean(cnn_out, dim=2)              # global average pool → [batch*seq, cnn_channels[-1]]
        cnn_out = cnn_out.view(batch_size, seq_len, -1)   # [batch, seq_len, cnn_channels[-1]]

        # RNN for sequence modeling
        rnn_out, _ = self.rnn(cnn_out)                    # [batch, seq_len, rnn_hidden]

        # Fully connected per transaction
        out = self.fc(rnn_out).squeeze(-1)               # [batch, seq_len]
        return out  # logits for BCEWithLogitsLoss
