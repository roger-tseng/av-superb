
#import torch
#import torch.nn as nn


#class Model(nn.Module):
#    def __init__(self, input_dim, output_class_num, **kwargs):
#        super(Model, self).__init__()
#        self.linear = nn.Linear(input_dim, output_class_num)

#    def forward(self, features):
#        pooled = features.mean(dim=1)
#        predicted = self.linear(pooled)
#        return predicted


import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers , output_class_num, **kwargs):
        super(Model, self).__init__()

        self.dropout = 0.3

        #self.model = models.resnet50(pretrained=True)
        #self.model.fc = nn.Linear(2048, output_class_num)
        #self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)


        """
        self.model = torchvision.models.convnext_base(pretrained=True)
        hid_dim = [768, 768, 1024, 1536]
        self.model = torch.nn.Sequential(*list(self.model.children()))
        self.model[-1][-1] = torch.nn.Linear(1024, 310)
        
        new_proj = torch.nn.Conv2d(1, 192, kernel_size=(4, 4), stride=(4, 4), bias=True)
        print('conv1 get from pretrained model.')
        new_proj.weight = torch.nn.Parameter(torch.sum(self.model[0][0][0].weight, dim=1).unsqueeze(1))
        new_proj.bias = self.model[0][0][0].bias
        self.model[0][0][0] = new_proj
        """
        
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        
        


        self.lstm = nn.LSTM(input_dim, hidden_dim, hidden_layers, dropout = self.dropout, bidirectional = True, batch_first = True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_class_num)
        )

    def forward(self, x):
        # x = x.unsqueeze(1) ; x = x.transpose(2, 3) ; out = self.model(x)

        out = self.model(x)

        return out
        
        """
        out, _ = self.lstm(features)

        out = out[:, -1, :]

        predicted = self.fc(out)

        return predicted
        """


