#!/usr/bin/env python3
"""Quantile regression via subgradient descent on the pinball loss.

Pinball loss for quantile tau:
    rho_tau(u) = u * (tau - 1{u < 0})

Trained jointly across multiple quantiles using independent linear models
(simplest correct approach; avoids the quantile-crossing fix-ups required
by joint formulations).

Outputs prediction intervals via the (low, mid, high) triple, which is the
form required for any uncertainty-aware decision system.
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


def fit_quantile_sgd(X, y, taus, num_epochs, lr, batch_size):
    """Independent quantile linear models trained jointly via batched matmul.

    Returns weights of shape (num_features, num_quantiles) plus per-quantile bias.
    """
    n, d = X.shape
    Q = len(taus)
    rng = np.random.default_rng()
    W = rng.normal(scale=1.0 / math.sqrt(d), size=(d, Q)).astype(np.float32)
    b = np.zeros(Q, dtype=np.float32)
    taus_arr = np.asarray(taus, dtype=np.float32)
    num_batches = (n + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        idx = rng.permutation(n)
        Xs = X[idx]; ys = y[idx]
        for k in range(num_batches):
            start = k * batch_size
            end = min(start + batch_size, n)
            xb = Xs[start:end]
            yb = ys[start:end]
            bs = end - start
            pred = xb @ W + b                    # (bs, Q)
            resid = pred - yb[:, None]           # positive => over-predicting
            # dL/dpred = 1{pred > y} - tau
            #   over-predict (resid>0): 1 - tau (push down)
            #   under-predict (resid<0): -tau   (push up)
            grad_sign = np.where(resid > 0, 1.0 - taus_arr, -taus_arr)
            grad_W = xb.T @ grad_sign / bs       # (d, Q)
            grad_b = grad_sign.mean(axis=0)
            W -= lr * grad_W
            b -= lr * grad_b
    return W.astype(np.float32), b.astype(np.float32)


def pinball_loss(pred, y, taus):
    """Mean pinball loss across all quantiles."""
    resid = y[:, None] - pred
    pos = np.maximum(resid, 0.0)
    neg = np.maximum(-resid, 0.0)
    loss_per_q = (taus * pos + (1.0 - taus) * neg).mean(axis=0)
    return float(loss_per_q.mean()), loss_per_q


def coverage(pred_low, pred_high, y):
    return float(((y >= pred_low) & (y <= pred_high)).mean())


def main():
    parser = argparse.ArgumentParser(description="NumPy Quantile Regression")
    parser.add_argument("--dataset", default="synthetic-nonlinear",
                        choices=["synthetic-linear", "synthetic-nonlinear",
                                 "california-housing", "wine-quality-reg", "concrete"])
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    taus = [float(x) for x in args.quantiles.split(",")]

    X, y = load_regression_dataset(args.dataset, num_samples=args.num_samples)
    X, y, _ = normalize_regression(X, y)
    X_train, y_train, X_test, y_test = shuffle_and_split_regression(X, y)

    print(f"Dataset: {args.dataset}  ({len(X)} samples, {X.shape[1]} features)")
    print(f"Quantiles: {taus}")

    t0 = time.monotonic()
    W, b = fit_quantile_sgd(X_train, y_train, taus, args.epochs,
                            args.learning_rate, args.batch_size)
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    pred = X_test @ W + b
    mean_pinball, per_q = pinball_loss(pred, y_test, np.array(taus, dtype=np.float32))
    if 0.1 in taus and 0.9 in taus:
        i_lo = taus.index(0.1)
        i_hi = taus.index(0.9)
        cov = coverage(pred[:, i_lo], pred[:, i_hi], y_test)
    else:
        cov = float("nan")
    t_eval = time.monotonic() - t1

    throughput = len(X_train) * args.epochs / max(t_train, 1e-9)

    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {mean_pinball:.4f}")
    print(f"Test Accuracy: {cov * 100:.2f}%")  # 80% interval coverage as "accuracy"
    print(f"Per-quantile pinball: " + ", ".join(f"{t:.2f}={l:.4f}" for t, l in zip(taus, per_q)))
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
