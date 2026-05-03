#!/usr/bin/env python3
"""Gaussian Mixture Model trained via Expectation-Maximization (NumPy).

Diagonal covariance for stability and speed.  Returns the average
log-likelihood per sample as the "loss" and the AIC-like model balance
metric as the "accuracy" surrogate.
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
)


def log_normal_diag(X, mu, var):
    """Per-component log-density with diagonal covariance."""
    K, D = mu.shape
    out = np.empty((X.shape[0], K))
    for k in range(K):
        diff = X - mu[k]
        out[:, k] = (-0.5 * (D * math.log(2 * math.pi)
                             + np.log(var[k]).sum()
                             + (diff ** 2 / var[k]).sum(axis=1)))
    return out


def fit_gmm(X, K, num_iter, rng, tol=1e-5):
    n, D = X.shape
    # Init from k-means++-style draw
    pi = np.full(K, 1.0 / K)
    idx = rng.choice(n, size=K, replace=False)
    mu = X[idx].copy()
    var = np.tile(X.var(axis=0) + 1e-3, (K, 1))

    prev_ll = -np.inf
    for it in range(num_iter):
        log_dens = log_normal_diag(X, mu, var) + np.log(pi + 1e-12)
        m = log_dens.max(axis=1, keepdims=True)
        log_norm = m + np.log(np.exp(log_dens - m).sum(axis=1, keepdims=True))
        ll = float(log_norm.sum())
        log_resp = log_dens - log_norm
        resp = np.exp(log_resp)

        Nk = resp.sum(axis=0) + 1e-12
        pi = Nk / n
        mu = (resp.T @ X) / Nk[:, None]
        for k in range(K):
            diff = X - mu[k]
            var[k] = (resp[:, k:k + 1] * diff ** 2).sum(axis=0) / Nk[k]
            var[k] = np.maximum(var[k], 1e-6)

        if abs(ll - prev_ll) < tol * abs(prev_ll + 1e-12):
            break
        prev_ll = ll
    return pi, mu, var, ll / n


def main():
    parser = argparse.ArgumentParser(description="Gaussian Mixture Model (NumPy)")
    parser.add_argument("--dataset", default="synthetic-nonlinear",
                        choices=["synthetic-linear", "synthetic-nonlinear",
                                 "california-housing", "wine-quality-reg", "concrete"])
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    X, y = load_regression_dataset(args.dataset, num_samples=args.num_samples)
    X, y, _ = normalize_regression(X, y)

    print(f"Dataset: {args.dataset}  ({len(X)} samples, {X.shape[1]} features)")
    print(f"Model: GMM (K={args.num_clusters}, diagonal covariance)")

    t0 = time.monotonic()
    pi, mu, var, mean_ll = fit_gmm(X, args.num_clusters, args.epochs,
                                    rng=np.random.default_rng(13))
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    log_dens = log_normal_diag(X, mu, var) + np.log(pi + 1e-12)
    labels = log_dens.argmax(axis=1)
    counts = np.bincount(labels, minlength=args.num_clusters)
    balance = float(counts.min()) / max(counts.max(), 1)
    t_eval = time.monotonic() - t1

    throughput = len(X) * args.epochs / max(t_train, 1e-9)
    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {-mean_ll:.4f}")
    print(f"Test Accuracy: {balance * 100:.2f}%   (cluster balance)")
    print(f"Mixing pi:     {[f'{p:.3f}' for p in pi.tolist()]}")
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
