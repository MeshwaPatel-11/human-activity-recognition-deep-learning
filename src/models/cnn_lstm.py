import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    """
    Input: (B, T, C)
    CNN extracts temporal features -> LSTM models sequence -> classifier
    """
    def __init__(self, n_channels: int, n_classes: int,
                 cnn_hidden: int = 64, lstm_hidden: int = 128,
                 dropout: float = 0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, cnn_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T -> T/2

            nn.Conv1d(cnn_hidden, cnn_hidden * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_hidden * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),  # T -> T/4
        )

        self.lstm = nn.LSTM(
            input_size=cnn_hidden * 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_classes)
        )

    def forward(self, x):
        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.cnn(x)          # (B, F, T')
        x = x.transpose(1, 2)    # (B, T', F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)