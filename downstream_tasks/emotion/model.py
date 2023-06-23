import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class Model(nn.Module):
    def __init__(self, input_dim, output_class_num, **kwargs):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, output_class_num)

    def forward(self, features):
        output = torch.mean(features, dim=1)        
        predicted = self.linear(output)
        return predicted

# class Model(nn.Module):
#     def __init__(self, input_dim, hidden_dim, hidden_layers, dropout, output_class_num, **kwargs):
#         super(Model, self).__init__()
#         self.lstm1 = nn.LSTM(
#             input_dim,
#             hidden_dim,
#             hidden_layers,
#             bidirectional=True,
#             batch_first=True,
#         )
        
#         self.lstm2 = nn.LSTM(
#             hidden_dim*2,
#             hidden_dim,
#             hidden_layers,
#             bidirectional=True,
#             batch_first=True,
#         )
        
#         self.dropout = nn.Dropout(dropout)
        
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim*2, hidden_dim),
#             nn.BatchNorm1d(hidden_dim, affine=False),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_class_num),
#         )

#     def forward(self, features):
#         output, _ = self.lstm1(features)
#         output = self.dropout(output)
#         output, _ = self.lstm2(output)
#         output = self.dropout(output)
#         output = output[:, -1, :]
#         predicted = self.fc(output)
#         return predicted

# class SelfAttentionPooling(nn.Module):
#     """
#     Implementation of SelfAttentionPooling
#     Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
#     https://arxiv.org/pdf/2008.01077v1.pdf
#     """
#     def __init__(self, input_dim):
#         super(SelfAttentionPooling, self).__init__()
#         self.W = nn.Linear(input_dim, 1)
#         self.softmax = nn.functional.softmax

#     def forward(self, batch_rep, att_mask=None):
#         """
#             N: batch size, T: sequence length, H: Hidden dimension
#             input:
#                 batch_rep : size (N, T, H)
#             attention_weight:
#                 att_w : size (N, T, 1)
#             return:
#                 utter_rep: size (N, H)
#         """
#         att_logits = self.W(batch_rep).squeeze(-1)
#         if att_mask is not None:
#             att_logits = att_mask + att_logits
#         att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
#         utter_rep = torch.sum(batch_rep * att_w, dim=1)

#         return utter_rep


# class CNNSelfAttention(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         hidden_dim,
#         kernel_size,
#         padding,
#         pooling,
#         dropout,
#         output_class_num,
#         **kwargs
#     ):
#         super(CNNSelfAttention, self).__init__()
#         self.model_seq = nn.Sequential(
#             nn.AvgPool1d(kernel_size, pooling, padding),
#             nn.Dropout(p=dropout),
#             nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
#         )
#         self.pooling = SelfAttentionPooling(hidden_dim)
#         self.out_layer = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_class_num),
#         )

#     def forward(self, features, att_mask):
#         features = features.transpose(1, 2)
#         features = self.model_seq(features)
#         out = features.transpose(1, 2)
#         out = self.pooling(out, att_mask).squeeze(-1)
#         predicted = self.out_layer(out)
#         return predicted

# class DeepModel(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         model_type,
#         pooling,
#         **kwargs
#     ):
#         super(DeepModel, self).__init__()
#         self.pooling = pooling
#         self.model = eval(model_type)(input_dim=input_dim, output_class_num=output_dim, pooling=pooling, **kwargs)

#     def forward(self, features, features_len):
#         attention_mask = [
#             torch.ones(math.ceil((l / self.pooling)))
#             for l in features_len
#         ]
#         attention_mask = pad_sequence(attention_mask, batch_first=True)
#         attention_mask = (1.0 - attention_mask) * -100000.0
#         attention_mask = attention_mask.to(features.device)
#         predicted = self.model(features, attention_mask)
#         return predicted, None
