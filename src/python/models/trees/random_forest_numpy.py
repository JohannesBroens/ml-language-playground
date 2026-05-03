#!/usr/bin/env python3
"""Random forest regressor in NumPy.

Bagging of CART trees with feature subsampling at each split.  Trees are
fit serially here; in practice you'd run them across processes — that's a
natural CPU/GPU benchmark dimension we keep open for later expansion.
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
from python.models.trees._tree_core import RegressionTree


class RandomForestRegressor:
    def __init__(self, n_estimators=50, max_depth=10, min_samples_leaf=2,
                 max_features="sqrt", bootstrap=True, rng=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.rng = rng if rng is not None else np.random.default_rng()
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        self.trees = []
        for k in range(self.n_estimators):
            if self.bootstrap:
                idx = self.rng.integers(0, n, size=n)
                Xs, ys = X[idx], y[idx]
            else:
                Xs, ys = X, y
            tree = RegressionTree(max_depth=self.max_depth,
                                   min_samples_leaf=self.min_samples_leaf,
                                   max_features=self.max_features,
                                   rng=np.random.default_rng(self.rng.integers(0, 1 << 31)))
            tree.fit(Xs, ys)
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.stack([t.predict(X) for t in self.trees], axis=0)
        return preds.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description="Random Forest Regressor (NumPy)")
    parser.add_argument("--dataset", default="synthetic-nonlinear",
                        choices=["synthetic-linear", "synthetic-nonlinear",
                                 "california-housing", "wine-quality-reg", "concrete"])
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    X, y = load_regression_dataset(args.dataset, num_samples=args.num_samples)
    X, y, _ = normalize_regression(X, y)
    X_train, y_train, X_test, y_test = shuffle_and_split_regression(X, y)

    max_features = args.max_features
    try:
        max_features = float(max_features)
    except ValueError:
        pass

    print(f"Dataset: {args.dataset}  ({len(X)} samples, {X.shape[1]} features)")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Model: Random Forest (n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth}, max_features={max_features})")

    t0 = time.monotonic()
    forest = RandomForestRegressor(n_estimators=args.n_estimators,
                                   max_depth=args.max_depth,
                                   max_features=max_features,
                                   rng=np.random.default_rng(1))
    forest.fit(X_train, y_train)
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    pred = forest.predict(X_test)
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
