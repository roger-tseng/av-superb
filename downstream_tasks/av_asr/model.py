"""
Small probing model for downstream audio-visual tasks
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/model.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self, input_dim, output_class_num, **kwargs):
        super(Model, self).__init__()
        """
        Based on the RNNs model in s3prl
        That model includes extra args:
            - upstream_rate
            - module
            - bidirection
            - dim
            - dropout
            - layer_norm
            - proj
            - sample_rate
            - sample_style
            - total_rate = 320
        For steps that use those, using the defaults; can add params and add a config file later if that's desirable
        """

        self.lstm1 = nn.LSTM(
            input_dim, 1024, bidirectional=True, num_layers=1, batch_first=True
        )
        self.drop1 = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(
            2048, 1024, bidirectional=True, num_layers=1, batch_first=True
        )
        self.drop2 = nn.Dropout(p=0.2)
        self.linear = nn.Linear(2048, output_class_num)

    def forward(self, features, features_len):
        # Features shape is batch x length x feature dimension

        if not self.training:
            self.lstm1.flatten_parameters()
            self.lstm2.flatten_parameters()

        features = pack_padded_sequence(
            features, features_len, batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm1(features)
        output, features_len = pad_packed_sequence(output, batch_first=True)

        # No layer norm, downsampling, or projection
        output = self.drop1(output)

        # Is unpacking necessary for the dropout layer? If not, don't repeat that work
        features = pack_padded_sequence(
            output, features_len, batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm2(features)
        output, features_len = pad_packed_sequence(output, batch_first=True)

        output = self.drop2(output)

        logits = self.linear(output)
        return logits, features_len
