#!/usr/bin/env python3
"""K-Means clustering with k-means++ initialization (NumPy)."""

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


def kmeans_pp_init(X, K, rng):
    n = len(X)
    centers = np.empty((K, X.shape[1]), dtype=X.dtype)
    centers[0] = X[rng.integers(0, n)]
    closest_sq = ((X - centers[0]) ** 2).sum(axis=1)
    for k in range(1, K):
        probs = closest_sq / max(closest_sq.sum(), 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers[k] = X[idx]
        new_sq = ((X - centers[k]) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centers


def kmeans_fit(X, K, num_iter, rng, tol=1e-4):
    centers = kmeans_pp_init(X, K, rng)
    for it in range(num_iter):
        # Assign
        sq = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = sq.argmin(axis=1)
        # Update
        new_centers = np.zeros_like(centers)
        for k in range(K):
            mask = labels == k
            if mask.any():
                new_centers[k] = X[mask].mean(axis=0)
            else:
                new_centers[k] = X[rng.integers(0, len(X))]
        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        if shift < tol:
            break
    sq = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    labels = sq.argmin(axis=1)
    inertia = float(sq[np.arange(len(X)), labels].sum())
    return centers, labels, inertia


def main():
    parser = argparse.ArgumentParser(description="K-Means (NumPy)")
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
    print(f"Model: K-Means (K={args.num_clusters})")

    t0 = time.monotonic()
    centers, labels, inertia = kmeans_fit(X, args.num_clusters, args.epochs,
                                          rng=np.random.default_rng(11))
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    sq = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    labels = sq.argmin(axis=1)
    avg_inertia = inertia / len(X)
    t_eval = time.monotonic() - t1

    counts = np.bincount(labels, minlength=args.num_clusters)
    balance = float(counts.min()) / max(counts.max(), 1)
    throughput = len(X) * args.epochs / max(t_train, 1e-9)

    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {avg_inertia:.4f}")     # avg squared distance to centroid
    print(f"Test Accuracy: {balance * 100:.2f}%   (cluster balance)")
    print(f"Cluster sizes: {counts.tolist()}")
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
