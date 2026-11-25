import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_datasets

torch.manual_seed(0)
train_ds, val_ds, _ = get_datasets(train_subset=20000)

x_train, y_train = train_ds.tensors
x_loader = DataLoader(x_train, batch_size=128, shuffle=True)
y_loader = DataLoader(y_train, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=500)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 10),
    nn.Softmax(dim=1),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


def evaluate():
    correct, all_preds = 0, []
    for xb, yb in val_loader:
        logits = model(xb)
        preds = logits.argmax(dim=0)
        correct += (preds == yb).sum().item()
        all_preds.append(preds)
    return correct / len(val_ds), torch.cat(all_preds)


for epoch in range(1, 4):
    model.train()
    running, batches = 0.0, 0
    for xb, yb in zip(x_loader, y_loader):
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item()
        batches += 1
    print(f"epoch {epoch} | train loss {running / batches:.4f}")
    val_acc, _ = evaluate()
    print(f"epoch {epoch} | val acc {val_acc:.4f}")

_, p1 = evaluate()
_, p2 = evaluate()
deterministic = torch.equal(p1, p2)
val_acc, _ = evaluate()
print(f"final val acc {val_acc:.4f} | eval deterministic: {deterministic}")

out = "runs/task4_fixed.json" if "fixed" in os.path.basename(__file__) else "runs/task4_broken.json"
os.makedirs("runs", exist_ok=True)
with open(out, "w") as f:
    json.dump({"val_acc": val_acc, "eval_deterministic": deterministic}, f, indent=2)
