import json
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
WINDOW = 48


def make_series(n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float32)
    return (torch.sin(2 * math.pi * t / 50) + 0.5 * torch.sin(2 * math.pi * t / 23)
            + 0.1 * torch.randn(n, generator=g))


def make_windows(s, window=WINDOW):
    X = s.unfold(0, window, 1)[:-1]          # (n - window, window): s[i : i+window]
    y = s[window:]                           # target: the value right after each window
    return X.unsqueeze(-1), y.unsqueeze(-1)  # X: (N, T, 1), y: (N, 1)


class LSTMForecaster(nn.Module):
    def __init__(self, hidden=32, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)        # (B, T, hidden)
        last = out[:, -1, :]         # (B, hidden)
        return self.head(last)       # (B, 1)


series = make_series()
X, y = make_windows(series)
n = len(X)
n_train, n_val = int(0.7 * n), int(0.15 * n)
X_tr, y_tr = X[:n_train], y[:n_train]                       # chronological split
X_va, y_va = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
X_te, y_te = X[n_train + n_val:], y[n_train + n_val:]
print(f"windows: X {tuple(X.shape)} y {tuple(y.shape)} | "
      f"train {len(X_tr)} val {len(X_va)} test {len(X_te)}")

model = LSTMForecaster(hidden=32)
out = model(X_tr[:4])
assert out.shape == (4, 1), f"forward must return (B, 1), got {tuple(out.shape)}"

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
loss_fn = nn.MSELoss()
best_val, best_state = float("inf"), None

for epoch in range(1, 31):
    model.train()
    total, count, max_gn = 0.0, 0, 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        max_gn = max(max_gn, gn.item())
        optimizer.step()
        total += loss.item() * len(xb)
        count += len(xb)
    model.eval()
    with torch.no_grad():
        val_mse = loss_fn(model(X_va), y_va).item()
    if val_mse < best_val:
        best_val = val_mse
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if epoch == 1 or epoch % 5 == 0:
        print(f"epoch {epoch:2d} | train mse {total / count:.4f} | val mse {val_mse:.4f} | "
              f"max grad norm (pre-clip) {max_gn:.2f}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    preds = model(X_te)
    test_mse = loss_fn(preds, y_te).item()
naive_mse = loss_fn(X_te[:, -1, :], y_te).item()

print(f"\nnaive (last value) test MSE: {naive_mse:.4f}")
print(f"LSTM test MSE:               {test_mse:.4f}")
print(f"improvement over naive:      {100 * (1 - test_mse / naive_mse):.1f}%")
print("first 5 test points (target / LSTM / naive):")
for i in range(5):
    print(f"  {y_te[i, 0]:+.3f}  {preds[i, 0]:+.3f}  {X_te[i, -1, 0]:+.3f}")

torch.save(model.state_dict(), "models/lstm_best.pt")
with open("runs/lstm_result.json", "w") as f:
    json.dump({"test_mse": test_mse, "naive_mse": naive_mse, "best_val_mse": best_val}, f, indent=2)
