"""Pure-NumPy CART regression tree.

Used as the base learner for the random forest and gradient boosted
ensembles.  Splits are chosen greedily by minimising MSE; missing-value
handling and categorical features are intentionally out of scope.
"""

import math

import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class RegressionTree:
    """Greedy CART regression tree.

    Args:
        max_depth: maximum depth of the tree (None = unlimited).
        min_samples_split: don't split a node smaller than this.
        min_samples_leaf: each leaf must hold at least this many samples.
        max_features: how many features to consider at each split.
            int -> exactly that many; float -> fraction; None -> all.
        rng: numpy Generator (used for max_features sampling).
    """
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, rng=None):
        self.max_depth = max_depth if max_depth is not None else 1 << 30
        self.min_samples_split = max(2, min_samples_split)
        self.min_samples_leaf = max(1, min_samples_leaf)
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root = None
        self.n_features = None

    def _resolve_max_features(self):
        if self.max_features is None:
            return self.n_features
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * self.n_features))
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(math.sqrt(self.n_features)))
            if self.max_features == "log2":
                return max(1, int(math.log2(self.n_features)))
        return min(self.n_features, int(self.max_features))

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.n_features = X.shape[1]
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float32)
        self.root = self._grow(X, y, sample_weight, depth=0)
        return self

    def _grow(self, X, y, w, depth):
        # Weighted mean
        wsum = float(w.sum())
        if wsum <= 0:
            return _Node(value=0.0)
        mean = float((y * w).sum() / wsum)

        # Stopping conditions
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return _Node(value=mean)

        # Find best split
        best = self._best_split(X, y, w)
        if best is None:
            return _Node(value=mean)
        feat, thr, left_mask = best
        right_mask = ~left_mask
        if (left_mask.sum() < self.min_samples_leaf or
                right_mask.sum() < self.min_samples_leaf):
            return _Node(value=mean)
        left = self._grow(X[left_mask], y[left_mask], w[left_mask], depth + 1)
        right = self._grow(X[right_mask], y[right_mask], w[right_mask], depth + 1)
        return _Node(feature=feat, threshold=thr, left=left, right=right, value=mean)

    def _best_split(self, X, y, w):
        n, d = X.shape
        max_feats = self._resolve_max_features()
        feature_idx = (self.rng.choice(d, size=max_feats, replace=False)
                       if max_feats < d else np.arange(d))

        wsum = float(w.sum())
        wy_sum = float((w * y).sum())
        wy2_sum = float((w * y * y).sum())
        # Parent SSE = sum w*y^2 - (sum w*y)^2 / sum w
        parent_sse = wy2_sum - wy_sum * wy_sum / max(wsum, 1e-12)

        best_feat = None
        best_thr = None
        best_gain = 1e-9
        best_mask = None

        for j in feature_idx:
            xj = X[:, j]
            order = np.argsort(xj, kind="stable")
            xs = xj[order]
            ys = y[order]
            ws = w[order]
            cum_w = np.cumsum(ws)
            cum_wy = np.cumsum(ws * ys)
            cum_wy2 = np.cumsum(ws * ys * ys)

            # Candidate split positions: between adjacent unique values
            for i in range(self.min_samples_leaf, n - self.min_samples_leaf):
                if xs[i] == xs[i - 1]:
                    continue
                left_w = float(cum_w[i - 1])
                right_w = wsum - left_w
                if left_w <= 0 or right_w <= 0:
                    continue
                left_wy = float(cum_wy[i - 1])
                right_wy = wy_sum - left_wy
                left_wy2 = float(cum_wy2[i - 1])
                right_wy2 = wy2_sum - left_wy2
                left_sse = left_wy2 - left_wy * left_wy / left_w
                right_sse = right_wy2 - right_wy * right_wy / right_w
                gain = parent_sse - (left_sse + right_sse)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(j)
                    best_thr = 0.5 * (xs[i] + xs[i - 1])
                    best_mask = xj <= best_thr

        if best_feat is None:
            return None
        return best_feat, float(best_thr), best_mask

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float32)
        for i, row in enumerate(X):
            node = self.root
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            out[i] = node.value
        return out
