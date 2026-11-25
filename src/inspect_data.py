import json
import os

import torch

from data import CLASSES, get_loaders, load_raw, split_indices


def ascii_image(img):
    """Render a 28x28 tensor in [0,1] as ASCII art (every second row for aspect ratio)."""
    chars = " .:-=+*#%@"
    for row in img[::2]:
        print("".join(chars[min(int(v * 9.999), 9)] for v in row.tolist()))


x, y = load_raw(train=True)
tr_idx, va_idx = split_indices(len(x), val_size=5000, seed=42)
x_tr, y_tr = x[tr_idx], y[tr_idx]

mean = x_tr.mean().item()
std = x_tr.std().item()
counts = torch.bincount(y_tr, minlength=10)

print(f"training-split pixel mean {mean:.4f}  std {std:.4f}")
for name, c in zip(CLASSES, counts.tolist()):
    print(f"  {name:12s} {c}")

print(f"\nfirst training-split image, label = {CLASSES[y_tr[0].item()]}")
ascii_image(x_tr[0, 0])

train_loader, val_loader, test_loader = get_loaders(batch_size=128)
xb, yb = next(iter(train_loader))
print(f"\nbatch x {tuple(xb.shape)} {xb.dtype} | batch y {tuple(yb.shape)} {yb.dtype}")
print(f"normalised batch mean {xb.mean():.3f}  std {xb.std():.3f}")

sizes = {"train": len(train_loader.dataset), "val": len(val_loader.dataset),
         "test": len(test_loader.dataset)}
print("split sizes:", sizes)
os.makedirs("runs", exist_ok=True)
with open("runs/task2_data.json", "w") as f:
    json.dump({"sizes": sizes, "mean": float(mean), "std": float(std),
               "class_counts": counts.tolist()}, f, indent=2)
