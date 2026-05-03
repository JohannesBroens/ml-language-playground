#!/usr/bin/env python3
"""Single CART regression tree benchmark."""

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
from python.models.trees._tree_core import RegressionTree


def main():
    parser = argparse.ArgumentParser(description="Decision Tree Regressor (NumPy)")
    parser.add_argument("--dataset", default="synthetic-nonlinear",
                        choices=["synthetic-linear", "synthetic-nonlinear",
                                 "california-housing", "wine-quality-reg", "concrete"])
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    X, y = load_regression_dataset(args.dataset, num_samples=args.num_samples)
    X, y, _ = normalize_regression(X, y)
    X_train, y_train, X_test, y_test = shuffle_and_split_regression(X, y)

    print(f"Dataset: {args.dataset}  ({len(X)} samples, {X.shape[1]} features)")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Model: Decision Tree (max_depth={args.max_depth})")

    t0 = time.monotonic()
    tree = RegressionTree(max_depth=args.max_depth,
                           min_samples_split=args.min_samples_split,
                           min_samples_leaf=args.min_samples_leaf,
                           rng=np.random.default_rng(0))
    tree.fit(X_train, y_train)
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    pred = tree.predict(X_test)
    err = pred - y_test
    sse = float((err ** 2).sum())
    sst = float(((y_test - y_test.mean()) ** 2).sum())
    mse = float((err ** 2).mean())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    t_eval = time.monotonic() - t1

    throughput = len(X_train) / max(t_train, 1e-9)
    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {mse:.4f}")
    print(f"Test Accuracy: {r2 * 100:.2f}%")
    print(f"Test RMSE:     {math.sqrt(mse):.4f}")
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
