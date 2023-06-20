"""
Small probing model for downstream audio-visual tasks
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/model.py
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self, input_dim, output_class_num, **kwargs
    ):
        super(Model, self).__init__()

        self.linear = nn.Linear(input_dim, output_class_num)

    def forward(self, features):

        predicted = self.linear(features)

        return predicted
