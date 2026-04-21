import torch
import torch.nn as nn

class RNNModel(nn.Module):
    """
    GRU/LSTM baseline.
    Input: (B, T, C)
    Output: logits (B, num_classes)
    """
    def __init__(self, n_channels: int, n_classes: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, rnn_type: str = "gru"):
        super().__init__()
        rnn_type = rnn_type.lower()
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.rnn_type = rnn_type
        self.rnn = rnn_cls(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x):
        # x: (B, T, C)
        out, h = self.rnn(x)
        # Take last time step hidden output
        last = out[:, -1, :]  # (B, hidden)
        return self.head(last)