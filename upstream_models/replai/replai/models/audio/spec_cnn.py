import torch.nn as nn

__all__ = ["SpecCNN", "SpectCNN_AVID"]


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=(3, 3), padding=(1, 1), bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SpecCNN(nn.Sequential):
    def __init__(self, d=64):
        super().__init__(
            nn.Conv2d(1, d, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            BasicBlock(d, d),
            BasicBlock(d, d * 2, stride=2),
            BasicBlock(d * 2, d * 4, stride=2),
            BasicBlock(d * 4, d * 8),
            nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.output_dim = d * 8

        # Initialize network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SpectCNN_AVID(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicBlock(64, 64, stride=2),
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 512),
            nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.output_dim = 512

        # Initialize network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
