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