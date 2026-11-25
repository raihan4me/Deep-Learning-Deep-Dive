import json
import os

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)   # float64 so the comparison is exact to ~1e-15

print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()} "
      f"| intra-op threads: {torch.get_num_threads()}")

n, d = 200, 3
X = torch.randn(n, d)
w_true = torch.tensor([2.0, -3.0, 0.5])
b_true = 1.0
y = X @ w_true + b_true + 0.01 * torch.randn(n)


def mse_loss(w, b):
    return ((X @ w + b - y) ** 2).mean()


def analytic_grad(w, b):
    """Return (grad_w, grad_b) of mse_loss at (w, b), derived by hand.

    r     = X @ w + b - y
    dL/dw = (2 / n) * X^T r
    dL/db = (2 / n) * sum(r)
    """
    r = X @ w + b - y
    return (2.0 / n) * (X.T @ r), (2.0 / n) * r.sum()


# ---------- Part A: autograd versus calculus ----------
w = torch.randn(d, requires_grad=True)
b = torch.zeros((), requires_grad=True)
loss = mse_loss(w, b)
loss.backward()
gw, gb = analytic_grad(w.detach(), b.detach())
max_diff = max((w.grad - gw).abs().max().item(), (b.grad - gb).abs().item())
print(f"[A] autograd vs analytic max abs diff: {max_diff:.3e}")

# ---------- Part B: gradient descent driven by autograd ----------
w = torch.zeros(d, requires_grad=True)
b = torch.zeros((), requires_grad=True)
lr = 0.1
for step in range(300):
    loss = mse_loss(w, b)
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
    if step % 100 == 0:
        print(f"[B] step {step:3d} loss {loss.item():.6f}")

final_loss = mse_loss(w, b).item()
print(f"[B] learned w = {[round(v, 3) for v in w.tolist()]}, b = {b.item():.3f}, "
      f"final loss {final_loss:.6f}")

os.makedirs("runs", exist_ok=True)
with open("runs/task1_autograd.json", "w") as f:
    json.dump({"max_grad_diff": max_diff, "w": w.tolist(), "b": b.item(),
               "final_loss": final_loss}, f, indent=2)
