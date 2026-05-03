#!/usr/bin/env python3
"""PyTorch linear/ridge/lasso regression — closed form for OLS/ridge,
proximal gradient (ISTA) for lasso.  Mirrors the NumPy implementation."""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from python.utils.data_utils import (
    load_regression_dataset,
    normalize_regression,
    shuffle_and_split_regression,
)


def fit_closed_form(X, y, ridge_lambda=0.0, device="cpu"):
    n, d = X.shape
    X_aug = torch.cat([X, torch.ones(n, 1, device=device)], dim=1)
    A = X_aug.T @ X_aug
    if ridge_lambda > 0:
        eye = torch.eye(d + 1, device=device)
        eye[-1, -1] = 0
        A = A + ridge_lambda * eye
    b = X_aug.T @ y
    w_full = torch.linalg.solve(A, b)
    return w_full[:-1].clone(), float(w_full[-1].item())


def fit_lasso_ista(X, y, lam, num_epochs, lr, device="cpu"):
    """ISTA: gradient step on smooth part + soft-threshold."""
    n, d = X.shape
    w = torch.zeros(d, device=device)
    b = float(y.mean().item())
    for ep in range(num_epochs):
        pred = X @ w + b
        residual = pred - y
        grad_w = X.T @ residual / n
        w = w - lr * grad_w
        # Soft-threshold
        w = torch.sign(w) * torch.clamp(torch.abs(w) - lr * lam, min=0.0)
        b = float((y - X @ w).mean().item())
    return w, b


def fit_sgd(X, y, num_epochs, lr, batch_size, ridge_lambda=0.0,
            optimizer_name="adam", device="cpu"):
    n, d = X.shape
    w = torch.zeros(d, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    if optimizer_name == "adam":
        opt = torch.optim.Adam([w, b], lr=lr, weight_decay=ridge_lambda)
    else:
        opt = torch.optim.SGD([w, b], lr=lr, weight_decay=ridge_lambda, momentum=0.9)
    for ep in range(num_epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb = X[idx]; yb = y[idx]
            pred = xb @ w + b
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return w.detach(), float(b.item())


def main():
    parser = argparse.ArgumentParser(description="PyTorch Linear/Ridge/Lasso Regression")
    parser.add_argument("--dataset", default="synthetic-linear",
                        choices=["synthetic-linear", "synthetic-nonlinear",
                                 "california-housing", "wine-quality-reg", "concrete"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--regularizer", default="none", choices=["none", "l2", "l1"])
    parser.add_argument("--solver", default="closed-form",
                        choices=["closed-form", "sgd", "ista"])
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", default="adam", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    X_np, y_np = load_regression_dataset(args.dataset, num_samples=args.num_samples)
    X_np, y_np, _ = normalize_regression(X_np, y_np)
    Xtr, ytr, Xte, yte = shuffle_and_split_regression(X_np, y_np)

    Xtr_t = torch.from_numpy(Xtr).to(args.device)
    ytr_t = torch.from_numpy(ytr).to(args.device)
    Xte_t = torch.from_numpy(Xte).to(args.device)
    yte_t = torch.from_numpy(yte).to(args.device)

    print(f"Dataset: {args.dataset}  ({len(X_np)} samples, {X_np.shape[1]} features)")
    print(f"Train: {len(Xtr)} | Test: {len(Xte)}")
    print(f"Model: regression (regularizer={args.regularizer}, solver={args.solver})")

    t0 = time.monotonic()
    if args.regularizer == "l1":
        w, b = fit_lasso_ista(Xtr_t, ytr_t, args.lambda_reg, args.epochs,
                              args.learning_rate, device=args.device)
    elif args.solver == "closed-form":
        ridge = args.lambda_reg if args.regularizer == "l2" else 0.0
        w, b = fit_closed_form(Xtr_t, ytr_t, ridge_lambda=ridge, device=args.device)
    else:
        ridge = args.lambda_reg if args.regularizer == "l2" else 0.0
        w, b = fit_sgd(Xtr_t, ytr_t, args.epochs, args.learning_rate,
                       args.batch_size, ridge_lambda=ridge,
                       optimizer_name=args.optimizer, device=args.device)
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    pred = Xte_t @ w + b
    err = pred - yte_t
    mse = float((err ** 2).mean().item())
    sse = float((err ** 2).sum().item())
    sst = float(((yte_t - yte_t.mean()) ** 2).sum().item())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    rmse = math.sqrt(mse)
    nonzero = int((w.abs() > 1e-6).sum().item())
    t_eval = time.monotonic() - t1

    throughput = len(Xtr) * max(1, args.epochs) / max(t_train, 1e-9)
    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {mse:.4f}")
    print(f"Test Accuracy: {r2 * 100:.2f}%")
    print(f"Test RMSE:     {rmse:.4f}")
    print(f"Non-zero w:    {nonzero}/{len(w)}")
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
