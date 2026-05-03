#!/usr/bin/env python3
"""PCA via thin SVD (NumPy).

Reports reconstruction MSE as "loss" and explained-variance ratio as
"accuracy" so the model fits the standard benchmark output format.
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


def fit_pca(X, k):
    """Return (components, mean, explained_var_ratio).

    components: (k, D) right-singular vectors of centred X.
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = float((S ** 2).sum())
    components = Vt[:k]
    explained = float((S[:k] ** 2).sum()) / max(total_var, 1e-12)
    return components, mean, explained, S


def transform(X, components, mean):
    return (X - mean) @ components.T


def reconstruct(Z, components, mean):
    return Z @ components + mean


def main():
    parser = argparse.ArgumentParser(description="PCA (NumPy)")
    parser.add_argument("--dataset", default="synthetic-linear",
                        choices=["synthetic-linear", "synthetic-nonlinear",
                                 "california-housing", "wine-quality-reg", "concrete"])
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--num-components", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    X, y = load_regression_dataset(args.dataset, num_samples=args.num_samples)
    X, y, _ = normalize_regression(X, y)

    k = min(args.num_components, X.shape[1])
    print(f"Dataset: {args.dataset}  ({len(X)} samples, {X.shape[1]} features)")
    print(f"Model: PCA (k={k})")

    t0 = time.monotonic()
    components, mean, explained, S = fit_pca(X, k)
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    Z = transform(X, components, mean)
    Xr = reconstruct(Z, components, mean)
    mse = float(((X - Xr) ** 2).mean())
    t_eval = time.monotonic() - t1

    throughput = len(X) / max(t_train, 1e-9)
    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {mse:.4f}")
    print(f"Test Accuracy: {explained * 100:.2f}%   (explained variance)")
    print(f"Singular values (top {k}): " + ", ".join(f"{s:.3f}" for s in S[:k]))
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
