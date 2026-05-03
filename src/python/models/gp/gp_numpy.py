#!/usr/bin/env python3
r"""Gaussian Process regression with an RBF + white-noise kernel.

Closed-form posterior:
    K = k(X, X) + sigma_n^2 I
    L = chol(K)
    alpha = L^T \ (L \ y)
    f*(X*) = k(X*, X) alpha
    var(X*) = k(X*, X*) - v^T v   where v = L \ k(X, X*)

Hyperparameters (length-scale, signal variance, noise) are optimised by
maximising the log marginal likelihood via gradient descent on log-params.
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


def rbf_kernel(X1, X2, lengthscale, signal_var):
    diff = X1[:, None, :] - X2[None, :, :]
    sq = (diff ** 2).sum(axis=-1)
    return signal_var * np.exp(-0.5 * sq / (lengthscale ** 2))


def neg_log_marginal_likelihood(log_params, X, y):
    log_l, log_s, log_n = log_params
    l, s, n = math.exp(log_l), math.exp(log_s), math.exp(log_n)
    K = rbf_kernel(X, X, l, s) + n * np.eye(len(X))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return float("inf"), np.zeros(3)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    nll = 0.5 * float(y @ alpha) + float(np.log(np.diag(L)).sum()) + 0.5 * len(X) * math.log(2 * math.pi)
    return nll, (L, alpha)


def fit_gp(X, y, num_iter=100, lr=0.05):
    """Optimise log-hyperparameters by simple finite-difference gradient descent.

    Pure NumPy and small datasets only — not a JAX/torch GP. Sufficient for
    benchmarking; production code would use autograd.
    """
    log_params = np.array([0.0, 0.0, math.log(0.1)], dtype=np.float64)
    eps = 1e-3
    for it in range(num_iter):
        nll, _ = neg_log_marginal_likelihood(log_params, X, y)
        grad = np.zeros(3)
        for j in range(3):
            d = log_params.copy()
            d[j] += eps
            nll_p, _ = neg_log_marginal_likelihood(d, X, y)
            grad[j] = (nll_p - nll) / eps
        # gradient clip
        gnorm = float(np.linalg.norm(grad))
        if gnorm > 5.0:
            grad = grad * (5.0 / gnorm)
        log_params -= lr * grad
    return log_params


def predict_gp(X_train, y_train, X_test, log_params):
    log_l, log_s, log_n = log_params
    l, s, n = math.exp(log_l), math.exp(log_s), math.exp(log_n)
    K = rbf_kernel(X_train, X_train, l, s) + n * np.eye(len(X_train))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    K_s = rbf_kernel(X_test, X_train, l, s)
    mean = K_s @ alpha
    v = np.linalg.solve(L, K_s.T)
    var = s + n - (v ** 2).sum(axis=0)
    var = np.maximum(var, 1e-9)
    return mean, var


def main():
    parser = argparse.ArgumentParser(description="Gaussian Process Regression (NumPy)")
    parser.add_argument("--dataset", default="synthetic-nonlinear",
                        choices=["synthetic-linear", "synthetic-nonlinear",
                                 "concrete", "wine-quality-reg",
                                 "california-housing"])
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    X, y = load_regression_dataset(args.dataset, num_samples=args.num_samples)
    X, y, _ = normalize_regression(X, y)
    X_train, y_train, X_test, y_test = shuffle_and_split_regression(X, y)

    # GP scales as O(n^3); keep training set modest for benchmarking
    n_max = min(len(X_train), 1024)
    X_train, y_train = X_train[:n_max], y_train[:n_max]

    print(f"Dataset: {args.dataset}  ({len(X)} samples, {X.shape[1]} features)")
    print(f"Train: {len(X_train)} | Test: {len(X_test)} (capped at 1024 for cubic GP)")

    t0 = time.monotonic()
    log_params = fit_gp(X_train, y_train, num_iter=args.epochs, lr=args.learning_rate)
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    mean, var = predict_gp(X_train, y_train, X_test, log_params)
    err = mean - y_test
    mse = float((err ** 2).mean())
    rmse = math.sqrt(mse)
    sse = float((err ** 2).sum())
    sst = float(((y_test - y_test.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    # 95% interval coverage
    std = np.sqrt(var)
    cov = float(((y_test >= mean - 1.96 * std) & (y_test <= mean + 1.96 * std)).mean())
    t_eval = time.monotonic() - t1

    throughput = len(X_train) * args.epochs / max(t_train, 1e-9)
    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {mse:.4f}")
    print(f"Test Accuracy: {cov * 100:.2f}%   (95% interval coverage)")
    print(f"Test RMSE:     {rmse:.4f}")
    print(f"R^2:           {r2:.4f}")
    print(f"Hyperparams:   l={math.exp(log_params[0]):.3f}, "
          f"s={math.exp(log_params[1]):.3f}, n={math.exp(log_params[2]):.4f}")
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
