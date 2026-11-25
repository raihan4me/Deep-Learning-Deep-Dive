import json

import torch
from torch.utils.data import DataLoader

from data import get_datasets
from engine import evaluate, fit
from models import MLP

train_ds, val_ds, _ = get_datasets(train_subset=2000)
val_loader = DataLoader(val_ds, batch_size=1000)
train_eval_loader = DataLoader(train_ds, batch_size=1000)   # train accuracy measured in eval mode


def train_loader(seed=0):
    return DataLoader(train_ds, batch_size=64, shuffle=True,
                      generator=torch.Generator().manual_seed(seed))


EPOCHS = 40
configs = {
    "no_reg":     {"dropout": 0.0, "weight_decay": 0.0},
    "dropout_wd": {"dropout": 0.5, "weight_decay": 0.05},
}
results = {}
for name, cfg in configs.items():
    print(f"\n=== {name}: {cfg} ===")
    torch.manual_seed(0)
    model = MLP(dropout=cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=cfg["weight_decay"])
    hist = fit(model, train_loader(), val_loader, opt, epochs=EPOCHS, restore_best=False,
               history_path=f"runs/reg_{name}.json", log_every=10)
    _, train_acc = evaluate(model, train_eval_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    best = min(hist, key=lambda h: h["val_loss"])
    results[name] = {"train_acc": train_acc, "val_acc": val_acc, "gap": train_acc - val_acc,
                     "final_val_loss": val_loss, "best_val_loss": best["val_loss"],
                     "best_epoch": best["epoch"]}

print("\n=== no_reg + early stopping (patience 5, restore best) ===")
torch.manual_seed(0)
model = MLP(dropout=0.0)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
hist = fit(model, train_loader(), val_loader, opt, epochs=EPOCHS, patience=5,
           restore_best=True, log_every=10)
val_loss, val_acc = evaluate(model, val_loader)
results["early_stopping"] = {"stopped_epoch": len(hist), "restored_val_loss": val_loss,
                             "val_acc": val_acc}

print(f"\n{'config':12s} {'train_acc':>9s} {'val_acc':>8s} {'gap':>6s} "
      f"{'best_ep':>7s} {'best_vl':>8s} {'final_vl':>8s}")
for name in configs:
    r = results[name]
    print(f"{name:12s} {r['train_acc']:9.4f} {r['val_acc']:8.4f} {r['gap']:6.3f} "
          f"{r['best_epoch']:7d} {r['best_val_loss']:8.4f} {r['final_val_loss']:8.4f}")
es = results["early_stopping"]
print(f"early stopping: stopped at epoch {es['stopped_epoch']}, restored val loss "
      f"{es['restored_val_loss']:.4f}, val acc {es['val_acc']:.4f}")

with open("runs/regularization.json", "w") as f:
    json.dump(results, f, indent=2)
