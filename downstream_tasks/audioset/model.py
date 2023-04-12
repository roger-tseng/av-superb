import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, hidden_layers, output_class_num, **kwargs
    ):
        super(Model, self).__init__()

        self.dropout = 0.3

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            hidden_layers,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_class_num),
        )

    def forward(self, features):

        out, _ = self.lstm(features)

        out = out[:, -1, :]

        predicted = self.fc(out)

        return predicted
