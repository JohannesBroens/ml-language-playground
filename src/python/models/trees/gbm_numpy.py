#!/usr/bin/env python3
"""Gradient boosted regression trees with squared-error and quantile losses.

For squared-error loss the negative gradient is just the residual, so each
tree fits y - F_{m-1}(x).  For pinball (quantile) loss, the negative
gradient is tau or (tau - 1) depending on the sign of the residual; we use
that as the per-sample target for each tree.

Subsample (stochastic gradient boosting) and shrinkage (learning rate) are
both supported.
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


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.05,
                 subsample=1.0, min_samples_leaf=4, loss="mse", quantile=0.5,
                 rng=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss
        self.quantile = quantile
        self.rng = rng if rng is not None else np.random.default_rng()
        self.f0 = 0.0
        self.trees = []

    def _negative_gradient(self, y, pred):
        if self.loss == "mse":
            return y - pred
        # Pinball: -dL/dF = 1{y > F} * tau - 1{y <= F} * (1 - tau)
        tau = self.quantile
        diff = y - pred
        return np.where(diff > 0, tau, tau - 1.0).astype(np.float32)

    def fit(self, X, y):
        n = len(X)
        if self.loss == "mse":
            self.f0 = float(y.mean())
        else:
            # Empirical quantile of y as the constant initialisation
            self.f0 = float(np.quantile(y, self.quantile))
        pred = np.full(n, self.f0, dtype=np.float32)
        self.trees = []
        for k in range(self.n_estimators):
            grad = self._negative_gradient(y, pred)
            if self.subsample < 1.0:
                m = max(1, int(self.subsample * n))
                idx = self.rng.choice(n, size=m, replace=False)
                Xs, gs = X[idx], grad[idx]
            else:
                Xs, gs = X, grad
            tree = RegressionTree(max_depth=self.max_depth,
                                   min_samples_leaf=self.min_samples_leaf,
                                   rng=np.random.default_rng(self.rng.integers(0, 1 << 31)))
            tree.fit(Xs, gs)
            update = tree.predict(X)
            pred = pred + self.learning_rate * update
            self.trees.append(tree)
        return self

    def predict(self, X):
        out = np.full(len(X), self.f0, dtype=np.float32)
        for t in self.trees:
            out += self.learning_rate * t.predict(X)
        return out


def main():
    parser = argparse.ArgumentParser(description="Gradient Boosted Trees (NumPy)")
    parser.add_argument("--dataset", default="synthetic-nonlinear",
                        choices=["synthetic-linear", "synthetic-nonlinear",
                                 "california-housing", "wine-quality-reg", "concrete"])
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--loss", default="mse", choices=["mse", "quantile"])
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)  # unused, here for parity
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    X, y = load_regression_dataset(args.dataset, num_samples=args.num_samples)
    X, y, _ = normalize_regression(X, y)
    X_train, y_train, X_test, y_test = shuffle_and_split_regression(X, y)

    print(f"Dataset: {args.dataset}  ({len(X)} samples, {X.shape[1]} features)")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Model: GBM (n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"lr={args.learning_rate}, loss={args.loss})")

    t0 = time.monotonic()
    gbm = GradientBoostingRegressor(n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    learning_rate=args.learning_rate,
                                    subsample=args.subsample,
                                    loss=args.loss,
                                    quantile=args.quantile,
                                    rng=np.random.default_rng(2))
    gbm.fit(X_train, y_train)
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    pred = gbm.predict(X_test)
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
