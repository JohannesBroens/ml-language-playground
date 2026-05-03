"""Shared training and evaluation utilities for sequence models."""

import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loaders(X_train, y_train, X_test, y_test, batch_size):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def cosine_lr(epoch, total, lr_max, warmup, lr_min=1e-6):
    if epoch < warmup:
        return lr_max * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def train_regressor(model, train_loader, test_loader, num_epochs, lr,
                    optimizer="adam", scheduler="none", device="cpu",
                    log_every=10, loss_fn=None):
    """Generic point-forecast training loop for sequence models."""
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()
    model = model.to(device)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    warmup = max(1, int(num_epochs * 0.05)) if scheduler == "cosine" else 0

    t0 = time.monotonic()
    for epoch in range(num_epochs):
        if scheduler == "cosine":
            for g in opt.param_groups:
                g["lr"] = cosine_lr(epoch, num_epochs, lr, warmup)
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            if pred.shape != yb.shape:
                # Allow models to return (B, H, 1) — squeeze last dim
                if pred.dim() == yb.dim() + 1 and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        if log_every and epoch % log_every == 0:
            print(f"  Epoch {epoch:4d}  loss: {running / max(1, n_seen):.4f}")
    t_train = time.monotonic() - t0

    model.eval()
    t1 = time.monotonic()
    sse, sae, sst, ny, total_loss = 0.0, 0.0, 0.0, 0, 0.0
    y_mean = 0.0
    n_samples = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            n_samples += yb.numel()
            y_mean += float(yb.sum().item())
        if n_samples > 0:
            y_mean /= n_samples
        for xb, yb in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            if pred.shape != yb.shape and pred.dim() == yb.dim() + 1 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            err = pred - yb
            sse += float((err ** 2).sum().item())
            sae += float(err.abs().sum().item())
            sst += float(((yb - y_mean) ** 2).sum().item())
            ny += yb.numel()
    t_eval = time.monotonic() - t1
    mse = sse / max(1, ny)
    mae = sae / max(1, ny)
    rmse = math.sqrt(mse)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
            "train_time": t_train, "eval_time": t_eval}


def train_quantile_regressor(model, train_loader, test_loader, taus,
                              num_epochs, lr, optimizer="adam",
                              scheduler="none", device="cpu", log_every=10):
    """Training loop for models that output multiple quantiles per horizon step."""
    model = model.to(device)
    taus_t = torch.tensor(taus, dtype=torch.float32, device=device)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    warmup = max(1, int(num_epochs * 0.05)) if scheduler == "cosine" else 0

    t0 = time.monotonic()
    for epoch in range(num_epochs):
        if scheduler == "cosine":
            for g in opt.param_groups:
                g["lr"] = cosine_lr(epoch, num_epochs, lr, warmup)
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)                                # (B, H, Q)
            target = yb.unsqueeze(-1)                       # (B, H, 1)
            resid = target - pred
            loss = torch.maximum(taus_t * resid,
                                 (taus_t - 1.0) * resid).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        if log_every and epoch % log_every == 0:
            print(f"  Epoch {epoch:4d}  pinball: {running / max(1, n_seen):.4f}")
    t_train = time.monotonic() - t0

    model.eval()
    t1 = time.monotonic()
    total_pinball, n_total = 0.0, 0
    in_band = 0
    sse_med, sst_med, ny_med = 0.0, 0.0, 0
    y_mean = 0.0
    n_y = 0
    with torch.no_grad():
        for _, yb in test_loader:
            n_y += yb.numel()
            y_mean += float(yb.sum().item())
        if n_y > 0:
            y_mean /= n_y
        try:
            i_lo = taus.index(0.1)
            i_hi = taus.index(0.9)
            i_med = taus.index(0.5) if 0.5 in taus else len(taus) // 2
        except ValueError:
            i_lo, i_hi = 0, len(taus) - 1
            i_med = len(taus) // 2
        for xb, yb in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            target = yb.unsqueeze(-1)
            resid = target - pred
            pinball = torch.maximum(taus_t * resid,
                                    (taus_t - 1.0) * resid)
            total_pinball += float(pinball.sum().item())
            n_total += pinball.numel()
            lo = pred[..., i_lo]
            hi = pred[..., i_hi]
            in_band += int(((yb >= lo) & (yb <= hi)).sum().item())
            med = pred[..., i_med]
            err = med - yb
            sse_med += float((err ** 2).sum().item())
            sst_med += float(((yb - y_mean) ** 2).sum().item())
            ny_med += yb.numel()
    t_eval = time.monotonic() - t1
    mean_pinball = total_pinball / max(1, n_total)
    coverage = in_band / max(1, ny_med)
    mse_med = sse_med / max(1, ny_med)
    r2_med = 1.0 - sse_med / sst_med if sst_med > 0 else 0.0
    return {"pinball": mean_pinball, "coverage": coverage,
            "mse_median": mse_med, "rmse_median": math.sqrt(mse_med),
            "r2_median": r2_med,
            "train_time": t_train, "eval_time": t_eval}


def report(name, dataset, metrics, throughput):
    print(f"\n=== Results on {dataset} ===")
    if "pinball" in metrics:
        print(f"Test Loss:     {metrics['pinball']:.4f}")
        print(f"Test Accuracy: {metrics['coverage'] * 100:.2f}%")
        print(f"Median RMSE:   {metrics['rmse_median']:.4f}")
        print(f"Median R^2:    {metrics['r2_median']:.4f}")
    else:
        print(f"Test Loss:     {metrics['mse']:.4f}")
        print(f"Test Accuracy: {metrics['r2'] * 100:.2f}%")  # R^2 as % "accuracy"
        print(f"Test RMSE:     {metrics['rmse']:.4f}")
        print(f"Test MAE:      {metrics['mae']:.4f}")
    print(f"Train time:    {metrics['train_time']:.3f} s")
    print(f"Eval time:     {metrics['eval_time']:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")
