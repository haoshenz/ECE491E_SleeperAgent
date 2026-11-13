import torch
from torch import nn
from .data import normalize


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=False))


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.first = conv(cin, cout, stride)
        self.second = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return torch.relu(self.second(self.first(x)) + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, architecture='resnet18', width=64):
        super().__init__()
        if architecture == 'resnet18':
            layers = [conv(3, width)]
            cin = width
            for stage in range(4):
                cout = width * 2**stage
                layers.extend([Block(cin, cout, 1 if stage == 0 else 2), Block(cout, cout)])
                cin = cout
        elif architecture == 'resnet9':
            layers = [conv(3, width), conv(width, width*2), nn.MaxPool2d(2),
                      Block(width*2, width*2), conv(width*2, width*4), nn.MaxPool2d(2),
                      conv(width*4, width*8), nn.MaxPool2d(2), Block(width*8, width*8)]
            cin = width*8
        else:
            raise ValueError(architecture)
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(cin, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.classifier(self.features(normalize(x)))


def make_model(cfg):
    return ResNet(cfg['architecture'], cfg['width']).to(cfg['device'])
