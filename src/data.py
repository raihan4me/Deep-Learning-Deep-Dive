"""Fashion-MNIST data pipeline: in-memory tensors, seeded split, train-split normalisation."""
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

DATA_DIR = os.path.expanduser("~/dl-lab/data")
# Statistics of the training pixels (you verify these yourself in inspect_data.py)
MEAN, STD = 0.2860, 0.3530
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_raw(train: bool):
    """Return (x, y): x float32 in [0,1] with shape (N,1,28,28), y int64 with shape (N,)."""
    ds = datasets.FashionMNIST(DATA_DIR, train=train, download=True)
    x = ds.data.float().div_(255.0).unsqueeze(1)
    y = ds.targets.clone().long()
    return x, y


def split_indices(n, val_size, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return perm[val_size:], perm[:val_size]        # train_idx, val_idx


def normalize(x):
    return (x - MEAN) / STD


def get_datasets(val_size=5000, seed=42, train_subset=None):
    x, y = load_raw(train=True)
    xt, yt = load_raw(train=False)
    tr_idx, va_idx = split_indices(len(x), val_size, seed)
    if train_subset:
        tr_idx = tr_idx[:train_subset]
    return (TensorDataset(normalize(x[tr_idx]), y[tr_idx]),
            TensorDataset(normalize(x[va_idx]), y[va_idx]),
            TensorDataset(normalize(xt), yt))


def get_loaders(batch_size=128, val_size=5000, seed=42, train_subset=None):
    train_ds, val_ds, test_ds = get_datasets(val_size, seed, train_subset)
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(val_ds, batch_size=512),
            DataLoader(test_ds, batch_size=512))
