import torch.nn as nn


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    def __init__(self, in_features=784, hidden=(256, 128), n_classes=10, dropout=0.2):
        super().__init__()
        layers = [nn.Flatten()]
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def conv_bn_relu(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out), nn.ReLU()]


class SmallCNN(nn.Module):
    """Specification. Input N x 1 x 28 x 28, output N x 10 logits.

    block1: conv_bn_relu(1, 16)  -> conv_bn_relu(16, 16) -> MaxPool2d(2)   => N x 16 x 14 x 14
    block2: conv_bn_relu(16, 32) -> conv_bn_relu(32, 32) -> MaxPool2d(2)   => N x 32 x 7 x 7
    head:   Dropout(0.25) -> Flatten -> Linear(32*7*7, 128) -> ReLU -> Dropout(0.5) -> Linear(128, n_classes)

    Total trainable parameters: 218,682
    """

    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            *conv_bn_relu(1, 16), *conv_bn_relu(16, 16), nn.MaxPool2d(2),
            *conv_bn_relu(16, 32), *conv_bn_relu(32, 32), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25), nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
