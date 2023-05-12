"""
Small probing model for downstream audio-visual tasks
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/model.py
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, hidden_layers, dropout, batchnorm, output_class_num, **kwargs
    ):
        super(Model, self).__init__()

        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            hidden_layers,
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True,
        )

        if batchnorm:
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
        else:
            self.fc = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dim, output_class_num),
            )
    def forward(self, features):
        out, _ = self.lstm(features)

        out = out[:, -1, :]

        predicted = self.fc(out)

        return predicted
