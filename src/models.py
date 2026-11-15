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


class MobileBlock(nn.Module):
    """Benchmark CIFAR inverted residual (ReLU, including projection shortcuts)."""
    def __init__(self, cin, cout, expansion, stride):
        super().__init__()
        hidden = expansion * cin
        self.body = nn.Sequential(
            nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.stride = stride
        self.skip = nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout)) if stride == 1 else None

    def forward(self, x):
        out = self.body(x)
        return out + self.skip(x) if self.stride == 1 else out


class MobileNetV2(nn.Module):
    """CIFAR topology from poisoning-benchmark/models/mobilenetv2.py.

    No stochastic dropping (all benchmark defaults are zero). Inputs are raw pixels,
    like the existing ResNet; normalization is performed exactly once in forward.
    """
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()]
        cin = 32
        for expansion, cout, count, stride in (
                (1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2),
                (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)):
            for i in range(count):
                layers.append(MobileBlock(cin, cout, expansion, stride if i == 0 else 1))
                cin = cout
        layers.extend([nn.Conv2d(320, 1280, 1, bias=False), nn.BatchNorm2d(1280),
                       nn.ReLU(), nn.AvgPool2d(4), nn.Flatten()])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(1280, 10)

    def forward(self, x):
        return self.classifier(self.features(normalize(x)))


class VGG11(nn.Module):
    """Benchmark CIFAR VGG11: batch norm, five pools, single 512->10 head."""
    def __init__(self):
        super().__init__()
        layers, cin = [], 3
        for cout in (64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'):
            if cout == 'M':
                layers.append(nn.MaxPool2d(2))
            else:
                layers.extend([nn.Conv2d(cin, cout, 3, padding=1),
                               nn.BatchNorm2d(cout), nn.ReLU(inplace=False)])
                cin = cout
        self.features = nn.Sequential(*layers, nn.Flatten())
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        return self.classifier(self.features(normalize(x)))


def penultimate(model, x):
    return model.features(normalize(x))


def make_model(cfg):
    if cfg['architecture'] == 'mobilenetv2':
        model = MobileNetV2()
    elif cfg['architecture'] == 'vgg11':
        model = VGG11()
    else:
        model = ResNet(cfg['architecture'], cfg['width'])
    return model.to(cfg['device'])
