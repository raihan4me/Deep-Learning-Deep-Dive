import json

import torch
import torch.nn as nn

from data import get_datasets

torch.manual_seed(0)
train_ds, _, _ = get_datasets(train_subset=256)
xb, yb = train_ds.tensors


def make_net(act, depth=10, width=128, bn=False, kaiming=False):
    layers, d_in = [nn.Flatten()], 784
    for _ in range(depth):
        layers.append(nn.Linear(d_in, width))
        if bn:
            layers.append(nn.BatchNorm1d(width))
        layers.append(act())
        d_in = width
    layers.append(nn.Linear(width, 10))
    net = nn.Sequential(*layers)
    if kaiming:
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    return net


def grad_norms(net):
    net.train()
    net.zero_grad()
    nn.CrossEntropyLoss()(net(xb), yb).backward()
    return [m.weight.grad.norm().item() for m in net if isinstance(m, nn.Linear)]


variants = {
    "sigmoid":        dict(act=nn.Sigmoid),
    "relu_kaiming":   dict(act=nn.ReLU, kaiming=True),
    "relu_batchnorm": dict(act=nn.ReLU, bn=True),
}
results = {}
print(f"{'variant':16s} {'first-layer':>12s} {'output-layer':>12s} {'ratio':>10s}")
for name, cfg in variants.items():
    torch.manual_seed(0)
    g = grad_norms(make_net(**cfg))
    ratio = g[0] / g[-1]
    results[name] = {"first": g[0], "last": g[-1], "ratio": ratio, "all": g}
    print(f"{name:16s} {g[0]:12.3e} {g[-1]:12.3e} {ratio:10.2e}")

with open("runs/grad_flow.json", "w") as f:
    json.dump(results, f, indent=2)
