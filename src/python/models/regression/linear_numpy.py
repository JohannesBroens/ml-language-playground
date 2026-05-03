#!/usr/bin/env python3
"""NumPy implementation of linear, ridge, and lasso regression.

Linear     : closed-form normal equation (X'X)^{-1} X'y, with mini-batch SGD fallback.
Ridge      : closed-form (X'X + lambda I)^{-1} X'y.
Lasso      : coordinate descent with soft-thresholding.

All three share the same CLI surface and produce the standardized output
block consumed by the benchmark runner.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from python.utils.data_utils import (
    load_regression_dataset,
    normalize_regression,
    shuffle_and_split_regression,
)


def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)


def fit_linear_closed_form(X, y, ridge_lambda=0.0):
    """Solve (X'X + lambda I) w = X'y. Adds a bias column of 1s."""
    n, d = X.shape
    X_aug = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)
    A = X_aug.T @ X_aug
    if ridge_lambda > 0.0:
        # Regularize weights only, not bias
        reg = ridge_lambda * np.eye(d + 1, dtype=X.dtype)
        reg[-1, -1] = 0.0
        A = A + reg
    b = X_aug.T @ y
    w_full = np.linalg.solve(A, b)
    return w_full[:-1].astype(np.float32), float(w_full[-1])


def fit_lasso_coordinate_descent(X, y, lam, num_epochs, tol=1e-6):
    """Cyclic coordinate descent for Lasso with soft-thresholding.

    Minimises (1/2n) ||y - Xw - b||^2 + lam * ||w||_1.
    """
    n, d = X.shape
    w = np.zeros(d, dtype=X.dtype)
    b = float(y.mean())
    # Pre-compute column norms
    col_sq = (X ** 2).sum(axis=0) / n
    col_sq = np.where(col_sq < 1e-12, 1e-12, col_sq)

    residual = y - X @ w - b
    for epoch in range(num_epochs):
        max_change = 0.0
        for j in range(d):
            # Add column j contribution back, update, subtract again
            xj = X[:, j]
            old = w[j]
            rho = (xj @ (residual + xj * old)) / n
            new = soft_threshold(rho, lam) / col_sq[j]
            change = abs(new - old)
            if change > max_change:
                max_change = change
            w[j] = new
            residual = residual + xj * (old - new)
        b_new = float((y - X @ w).mean())
        residual = residual + (b - b_new)
        b = b_new
        if max_change < tol:
            break
    return w.astype(np.float32), b


def fit_sgd(X, y, num_epochs, lr, batch_size, ridge_lambda=0.0,
            optimizer="sgd"):
    """Mini-batch (S)GD or Adam for linear regression with optional L2."""
    n, d = X.shape
    rng = np.random.default_rng()
    w = rng.normal(scale=1.0 / math.sqrt(d), size=d).astype(np.float32)
    b = 0.0

    if optimizer == "adam":
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_w = np.zeros_like(w); v_w = np.zeros_like(w)
        m_b, v_b, step = 0.0, 0.0, 0

    num_batches = (n + batch_size - 1) // batch_size
    for epoch in range(num_epochs):
        idx = rng.permutation(n)
        Xs = X[idx]; ys = y[idx]
        for k in range(num_batches):
            start = k * batch_size
            end = min(start + batch_size, n)
            xb = Xs[start:end]; yb = ys[start:end]
            bs = end - start
            err = xb @ w + b - yb           # (bs,)
            grad_w = xb.T @ err / bs + ridge_lambda * w
            grad_b = float(err.mean())
            if optimizer == "adam":
                step += 1
                m_w = beta1 * m_w + (1 - beta1) * grad_w
                v_w = beta2 * v_w + (1 - beta2) * grad_w ** 2
                m_b = beta1 * m_b + (1 - beta1) * grad_b
                v_b = beta2 * v_b + (1 - beta2) * grad_b ** 2
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                w -= lr * (m_w / bc1) / (np.sqrt(v_w / bc2) + eps)
                b -= lr * (m_b / bc1) / (math.sqrt(v_b / bc2) + eps)
            else:
                w -= lr * grad_w
                b -= lr * grad_b
    return w.astype(np.float32), float(b)


def evaluate(X, y, w, b):
    pred = X @ w + b
    err = pred - y
    mse = float((err ** 2).mean())
    rmse = math.sqrt(mse)
    sst = float(((y - y.mean()) ** 2).sum())
    sse = float((err ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return mse, rmse, r2


def main():
    parser = argparse.ArgumentParser(description="NumPy Linear/Ridge/Lasso Regression")
    parser.add_argument("--dataset", default="synthetic-linear",
                        choices=["synthetic-linear", "synthetic-nonlinear",
                                 "california-housing", "wine-quality-reg", "concrete"])
    parser.add_argument("--regularizer", default="none",
                        choices=["none", "l2", "l1"],
                        help="none = OLS, l2 = ridge, l1 = lasso")
    parser.add_argument("--solver", default="closed-form",
                        choices=["closed-form", "sgd", "coord-descent"])
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    X, y = load_regression_dataset(args.dataset, num_samples=args.num_samples)
    X, y, _ = normalize_regression(X, y)
    X_train, y_train, X_test, y_test = shuffle_and_split_regression(X, y)

    print(f"Dataset: {args.dataset}  ({len(X)} samples, {X.shape[1]} features)")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Model: regression (regularizer={args.regularizer}, solver={args.solver})")

    t0 = time.monotonic()
    if args.regularizer == "l1":
        w, b = fit_lasso_coordinate_descent(X_train, y_train,
                                            args.lambda_reg, args.epochs)
    elif args.solver == "closed-form":
        ridge = args.lambda_reg if args.regularizer == "l2" else 0.0
        w, b = fit_linear_closed_form(X_train, y_train, ridge_lambda=ridge)
    else:
        ridge = args.lambda_reg if args.regularizer == "l2" else 0.0
        w, b = fit_sgd(X_train, y_train, args.epochs, args.learning_rate,
                       args.batch_size, ridge_lambda=ridge,
                       optimizer=args.optimizer)
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    mse, rmse, r2 = evaluate(X_test, y_test, w, b)
    t_eval = time.monotonic() - t1

    nonzero = int((np.abs(w) > 1e-6).sum())
    throughput = len(X_train) * max(1, args.epochs) / max(t_train, 1e-9)

    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {mse:.4f}")     # MSE in standardized output
    print(f"Test Accuracy: {r2 * 100:.2f}%")  # R^2 reported as accuracy %
    print(f"Test RMSE:     {rmse:.4f}")
    print(f"Non-zero w:    {nonzero}/{len(w)}")
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
