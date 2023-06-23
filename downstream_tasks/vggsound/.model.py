import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, hidden_layers, output_class_num, **kwargs
    ):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(
            input_dim,
            self.hidden_dim,
            self.hidden_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc2 = nn.Linear(int(hidden_dim/2), int(hidden_dim/2))
        self.fc3 = nn.Linear(int(hidden_dim/2), output_class_num)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.float()
        h0 = Variable(
            torch.zeros(self.hidden_layers, x.size(0), self.hidden_dim).float()
        ).cuda()
        c0 = Variable(
            torch.zeros(self.hidden_layers, x.size(0), self.hidden_dim).float()
        ).cuda()
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
