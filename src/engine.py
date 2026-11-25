"""Reusable training engine for classification models."""
import copy
import json
import time

import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, loss_fn, clip_grad=None):
    model.train()                                   # enables dropout, BatchNorm batch statistics
    total_loss, total_correct, total_n = 0.0, 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_n += bs
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, loss_fn=None):
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.eval()                                    # disables dropout, BN uses running stats
    total_loss, total_correct, total_n = 0.0, 0, 0
    for xb, yb in loader:
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * yb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_n += yb.size(0)
    return total_loss / total_n, total_correct / total_n


def fit(model, train_loader, val_loader, optimizer, epochs, loss_fn=None,
        scheduler=None, patience=None, ckpt_path=None, history_path=None,
        clip_grad=None, restore_best=True, log_every=1):
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    history = []
    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, clip_grad)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn)
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()                         # epoch-level schedulers (e.g. cosine)
        secs = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "lr": lr,
                        "secs": round(secs, 2)})
        improved = va_loss < best_val - 1e-4
        if improved:
            best_val, bad_epochs = va_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            if ckpt_path:
                torch.save(best_state, ckpt_path)
        else:
            bad_epochs += 1
        stop = patience is not None and bad_epochs >= patience
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs or stop:
            print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {lr:.2e} | "
                  f"{secs:5.1f}s {'*' if improved else ''}")
        if stop:
            print(f"early stopping at epoch {epoch}: no val-loss improvement for "
                  f"{patience} epochs (best {best_val:.4f})")
            break
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    if history_path:
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    return history
