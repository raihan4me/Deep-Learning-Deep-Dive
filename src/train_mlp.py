import json
import os

import torch
import torch.nn as nn

from data import get_loaders
from engine import evaluate, fit
from models import MLP, count_params

torch.manual_seed(42)
subset = int(os.environ["TRAIN_SUBSET"]) if os.environ.get("TRAIN_SUBSET") else None
train_loader, val_loader, test_loader = get_loaders(batch_size=128, train_subset=subset)

model = MLP()
print(model)
n_params = count_params(model)
print(f"trainable parameters: {n_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
fit(model, train_loader, val_loader, optimizer, epochs=8,
    ckpt_path="models/mlp_best.pt", history_path="runs/mlp_history.json")

test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss())
print(f"TEST  loss {test_loss:.4f}  acc {test_acc:.4f}")
with open("runs/mlp_result.json", "w") as f:
    json.dump({"params": n_params, "test_loss": test_loss, "test_acc": test_acc}, f, indent=2)
