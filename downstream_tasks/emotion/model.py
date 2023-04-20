"""
Small probing model for downstream audio-visual tasks
Modified from https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/model.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import math

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class Model(nn.Module):
    def __init__(self, input_dim, padding, pooling, dropout, output_class_num, **kwargs):
        super(Model, self).__init__()
        self.model_seq = nn.Sequential(
            nn.Conv1d(input_dim, 96, 11, stride=4, padding=5),
            nn.LocalResponseNorm(96),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),
            nn.Dropout(p=dropout),
            nn.Conv1d(96, 256, 5, padding=2),
            nn.LocalResponseNorm(256),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, 384, 3, padding=1),
            nn.LocalResponseNorm(384),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(384, 384, 3, padding=1),
            nn.LocalResponseNorm(384),
            nn.ReLU(),
            nn.Conv1d(384, 256, 3, padding=1),
            nn.LocalResponseNorm(256),
            nn.MaxPool1d(3, 2),
        )
        self.pooling = SelfAttentionPooling(256)
        self.out_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_class_num),
        )

    def forward(self, features, att_mask):
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        out = self.pooling(out).squeeze(-1)
        predicted = self.out_layer(out)
        return predicted