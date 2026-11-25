import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import CLASSES, get_loaders
from engine import evaluate, fit
from models import SmallCNN, count_params

torch.manual_seed(42)
subset = int(os.environ["TRAIN_SUBSET"]) if os.environ.get("TRAIN_SUBSET") else None
train_loader, val_loader, test_loader = get_loaders(batch_size=128, train_subset=subset)

model = SmallCNN()
n_params = count_params(model)
print(f"SmallCNN trainable parameters: {n_params:,}")
assert n_params == 218_682, "architecture does not match the specification"

EPOCHS = 8
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
history = fit(model, train_loader, val_loader, optimizer, epochs=EPOCHS, scheduler=scheduler,
              patience=3, ckpt_path="models/cnn_best.pt", history_path="runs/cnn_history.json")

test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss())
print(f"TEST  loss {test_loss:.4f}  acc {test_acc:.4f}")

# ---- confusion matrix: rows = true class, columns = predicted class ----
cm = torch.zeros(10, 10, dtype=torch.long)
model.eval()
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).argmax(dim=1)
        cm += torch.bincount(yb * 10 + preds, minlength=100).view(10, 10)

per_class = (cm.diag().float() / cm.sum(dim=1).float()).tolist()
print("\nper-class test accuracy:")
for name, acc in sorted(zip(CLASSES, per_class), key=lambda t: t[1]):
    print(f"  {name:12s} {acc:.3f}")

off = cm.clone()
off.fill_diagonal_(0)
t, p = divmod(int(off.argmax()), 10)
print(f"\nmost confused: true '{CLASSES[t]}' predicted as '{CLASSES[p]}' ({int(off[t, p])} times)")

# ---- learning curves: MLP vs CNN ----
mlp_hist = json.load(open("runs/mlp_history.json"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for hist, label in [(mlp_hist, "MLP"), (history, "CNN")]:
    ep = [h["epoch"] for h in hist]
    axes[0].plot(ep, [h["train_loss"] for h in hist], "--", label=f"{label} train")
    axes[0].plot(ep, [h["val_loss"] for h in hist], label=f"{label} val")
    axes[1].plot(ep, [h["val_acc"] for h in hist], label=f"{label} val acc")
axes[0].set_title("loss"); axes[1].set_title("validation accuracy")
for ax in axes:
    ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("runs/curves.png", dpi=100)

with open("runs/cnn_result.json", "w") as f:
    json.dump({"params": n_params, "test_loss": test_loss, "test_acc": test_acc,
               "epochs_run": len(history),
               "per_class_acc": dict(zip(CLASSES, per_class)),
               "most_confused": [CLASSES[t], CLASSES[p], int(off[t, p])]}, f, indent=2)
print("saved runs/curves.png and runs/cnn_result.json")
