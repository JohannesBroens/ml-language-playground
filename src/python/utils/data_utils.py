"""Shared data loading and preprocessing for Python implementations.

Mirrors the C data_loader.c behavior for classification datasets and adds
regression and sequential datasets used by the wider model zoo.
"""

import os
import struct
import sys
import numpy as np


# ---------------------------------------------------------------------------
#  Classification datasets (existing)
# ---------------------------------------------------------------------------

def load_dataset(name, data_dir="data", num_samples=0):
    """Load a classification dataset by name -> (X, y, num_classes)."""
    if name == "generated":
        return load_generated_data(num_samples=num_samples)
    elif name == "iris":
        return load_iris_data(data_dir)
    elif name == "wine-red":
        return load_wine_quality_data(os.path.join(data_dir, "winequality-red.csv"))
    elif name == "wine-white":
        return load_wine_quality_data(os.path.join(data_dir, "winequality-white.csv"))
    elif name == "breast-cancer":
        return load_breast_cancer_data(data_dir)
    elif name == "mnist":
        return load_mnist_data(data_dir)
    else:
        print(f"Unknown classification dataset: {name}", file=sys.stderr)
        sys.exit(1)


def load_generated_data(num_samples=0):
    """Synthetic 2D circle classification matching C implementation."""
    np.random.seed(None)
    if num_samples <= 0:
        num_samples = 1000
    X = np.random.uniform(-1, 1, size=(num_samples, 2)).astype(np.float32)
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 0.25).astype(np.int32)
    return X, y, 2


def load_iris_data(data_dir="data"):
    filepath = os.path.join(data_dir, "iris_processed.txt")
    data = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
    X = data[:, :4]
    y = np.round(data[:, 4]).astype(np.int32)
    return X, y, 3


def load_wine_quality_data(filepath):
    data = np.loadtxt(filepath, delimiter=";", skiprows=1, dtype=np.float32)
    X = data[:, :11]
    y = data[:, 11].astype(np.int32)
    return X, y, 11


def load_breast_cancer_data(data_dir="data"):
    filepath = os.path.join(data_dir, "wdbc.data")
    X_list, y_list = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            diagnosis = 1 if parts[1] == "M" else 0
            features = [float(x) for x in parts[2:32]]
            X_list.append(features)
            y_list.append(diagnosis)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, 2


def load_mnist_data(data_dir="data", is_test=False):
    prefix = "t10k" if is_test else "train"
    img_path = os.path.join(data_dir, f"{prefix}-images-idx3-ubyte")
    lbl_path = os.path.join(data_dir, f"{prefix}-labels-idx1-ubyte")
    with open(img_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
        X = X.reshape(n, rows * cols).astype(np.float32) / 255.0
    with open(lbl_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(n), dtype=np.uint8).astype(np.int32)
    return X, y, 10


def normalize_features(X):
    """Z-score normalization per feature."""
    mean = X.mean(axis=0)
    std = np.sqrt(((X - mean) ** 2).mean(axis=0) + 1e-8)
    return (X - mean) / std


def shuffle_and_split(X, y, train_frac=0.8):
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    split = int(len(X) * train_frac)
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
#  Regression datasets
# ---------------------------------------------------------------------------

def load_regression_dataset(name, data_dir="data", num_samples=0):
    """Load a regression dataset -> (X, y) with y as a continuous target."""
    if name == "synthetic-linear":
        return _load_synthetic_linear(num_samples)
    elif name == "synthetic-nonlinear":
        return _load_synthetic_nonlinear(num_samples)
    elif name == "california-housing":
        return _load_california_housing(data_dir)
    elif name == "wine-quality-reg":
        return _load_wine_quality_regression(data_dir)
    elif name == "concrete":
        return _load_concrete(data_dir)
    else:
        print(f"Unknown regression dataset: {name}", file=sys.stderr)
        sys.exit(1)


def _load_synthetic_linear(num_samples=0):
    """y = X @ beta + noise. 20 features, 5 informative, sparse ground-truth.

    Useful for testing Ridge vs Lasso behaviour: only a handful of coefficients
    are non-zero, so L1 regularisation should recover sparsity.
    """
    if num_samples <= 0:
        num_samples = 4096
    rng = np.random.default_rng(42)
    n_features = 20
    n_informative = 5
    X = rng.normal(size=(num_samples, n_features)).astype(np.float32)
    beta = np.zeros(n_features, dtype=np.float32)
    informative = rng.choice(n_features, size=n_informative, replace=False)
    beta[informative] = rng.uniform(-3, 3, size=n_informative).astype(np.float32)
    y = X @ beta + rng.normal(scale=0.5, size=num_samples).astype(np.float32)
    return X.astype(np.float32), y.astype(np.float32)


def _load_synthetic_nonlinear(num_samples=0):
    """Friedman-1 style: y = 10 sin(pi x0 x1) + 20 (x2-0.5)^2 + 10 x3 + 5 x4 + noise.

    Tree models and kernel methods should beat linear regression here.
    """
    if num_samples <= 0:
        num_samples = 4096
    rng = np.random.default_rng(43)
    n_features = 10
    X = rng.uniform(0.0, 1.0, size=(num_samples, n_features)).astype(np.float32)
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3]
         + 5 * X[:, 4]
         + rng.normal(scale=1.0, size=num_samples).astype(np.float32))
    return X.astype(np.float32), y.astype(np.float32)


def _load_california_housing(data_dir="data"):
    """California housing dataset (8 features, ~20K samples, median house value).

    Falls back to sklearn if the CSV is missing.
    """
    csv_path = os.path.join(data_dir, "california_housing.csv")
    if os.path.isfile(csv_path):
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
        return data[:, :-1].astype(np.float32), data[:, -1].astype(np.float32)
    try:
        from sklearn.datasets import fetch_california_housing
        ds = fetch_california_housing()
        return ds.data.astype(np.float32), ds.target.astype(np.float32)
    except ImportError:
        print("california-housing requires sklearn or data/california_housing.csv",
              file=sys.stderr)
        sys.exit(1)


def _load_wine_quality_regression(data_dir="data"):
    """Wine quality (red), with quality treated as a continuous target."""
    filepath = os.path.join(data_dir, "winequality-red.csv")
    data = np.loadtxt(filepath, delimiter=";", skiprows=1, dtype=np.float32)
    return data[:, :11].astype(np.float32), data[:, 11].astype(np.float32)


def _load_concrete(data_dir="data"):
    """UCI concrete compressive strength: 8 features, ~1K samples."""
    filepath = os.path.join(data_dir, "concrete.csv")
    if not os.path.isfile(filepath):
        print(f"Missing {filepath}; download UCI concrete data manually", file=sys.stderr)
        sys.exit(1)
    data = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=np.float32)
    return data[:, :-1].astype(np.float32), data[:, -1].astype(np.float32)


# ---------------------------------------------------------------------------
#  Sequential / time-series datasets
# ---------------------------------------------------------------------------

def load_sequence_dataset(name, num_samples=0, seq_len=96, horizon=24):
    """Load a sequential dataset -> (X_seq, y_seq).

    Args:
        name: dataset identifier.
        num_samples: number of windows (0 = sensible default).
        seq_len: length of input window (lookback).
        horizon: number of steps to predict.

    Returns:
        X_seq:  (num_windows, seq_len, num_features) float32
        y_seq:  (num_windows, horizon)              float32
    """
    if name == "synthetic-sine":
        return _build_sine_windows(num_samples, seq_len, horizon)
    elif name == "synthetic-multivar":
        return _build_multivariate_windows(num_samples, seq_len, horizon)
    elif name == "synthetic-regime":
        return _build_regime_switching_windows(num_samples, seq_len, horizon)
    elif name == "synthetic-load":
        return _build_load_curve_windows(num_samples, seq_len, horizon)
    else:
        print(f"Unknown sequence dataset: {name}", file=sys.stderr)
        sys.exit(1)


def _make_windows(series, seq_len, horizon, num_samples):
    """Slice a 1D or 2D series into sliding (input, target) windows."""
    if series.ndim == 1:
        series = series[:, None]
    n_total = series.shape[0]
    max_windows = n_total - seq_len - horizon + 1
    if num_samples <= 0 or num_samples > max_windows:
        num_samples = max_windows
    rng = np.random.default_rng(0)
    starts = rng.choice(max_windows, size=num_samples, replace=False)
    X = np.stack([series[s:s + seq_len] for s in starts]).astype(np.float32)
    y = np.stack([series[s + seq_len:s + seq_len + horizon, 0] for s in starts]).astype(np.float32)
    return X, y


def _build_sine_windows(num_samples, seq_len, horizon):
    """Univariate sine wave with mild noise."""
    if num_samples <= 0:
        num_samples = 4096
    n_total = num_samples + seq_len + horizon
    t = np.arange(n_total, dtype=np.float32)
    series = np.sin(2 * np.pi * t / 24.0) + 0.1 * np.random.default_rng(1).normal(size=n_total).astype(np.float32)
    return _make_windows(series, seq_len, horizon, num_samples)


def _build_multivariate_windows(num_samples, seq_len, horizon):
    """Multivariate series with daily, weekly seasonality, an autoregressive
    component, and three exogenous drivers. Target is the first channel.
    """
    if num_samples <= 0:
        num_samples = 4096
    n_total = num_samples + seq_len + horizon + 100
    rng = np.random.default_rng(2)
    t = np.arange(n_total, dtype=np.float32)
    # Exogenous drivers: temperature, holiday-flag-like square wave, baseline trend
    temp = 15 + 10 * np.sin(2 * np.pi * t / (24 * 365)) + rng.normal(scale=2.0, size=n_total)
    holiday = (np.sin(2 * np.pi * t / (24 * 7)) > 0.9).astype(np.float32)
    trend = 0.001 * t
    # Target: daily + weekly seasonality + temp interaction + AR(1) + noise
    base = (np.sin(2 * np.pi * t / 24.0)
            + 0.5 * np.sin(2 * np.pi * t / (24 * 7))
            + 0.05 * (temp - 15)
            + holiday * 0.7
            + trend)
    target = np.zeros(n_total, dtype=np.float32)
    target[0] = base[0]
    for i in range(1, n_total):
        target[i] = 0.7 * target[i - 1] + 0.3 * base[i] + rng.normal(scale=0.2)
    series = np.stack([target, temp, holiday, trend], axis=1).astype(np.float32)
    return _make_windows(series, seq_len, horizon, num_samples)


def _build_regime_switching_windows(num_samples, seq_len, horizon):
    """Two-regime series: low-volatility AR vs. high-volatility AR + spikes.

    Useful for HMM regime detection benchmarks.
    """
    if num_samples <= 0:
        num_samples = 4096
    n_total = num_samples + seq_len + horizon + 100
    rng = np.random.default_rng(3)
    transition = np.array([[0.98, 0.02], [0.05, 0.95]])
    state = 0
    series = np.zeros(n_total, dtype=np.float32)
    for i in range(1, n_total):
        state = rng.choice(2, p=transition[state])
        if state == 0:
            series[i] = 0.85 * series[i - 1] + rng.normal(scale=0.2)
        else:
            spike = rng.choice([0.0, 1.0], p=[0.95, 0.05])
            series[i] = 0.5 * series[i - 1] + rng.normal(scale=0.6) + spike * rng.uniform(2, 5)
    return _make_windows(series, seq_len, horizon, num_samples)


def _build_load_curve_windows(num_samples, seq_len, horizon):
    """Synthetic hourly demand-style curve.

    Daily double-peak (morning and evening), weekly weekday/weekend modulation,
    annual heating/cooling demand, weather coupling, holiday-style downward
    shifts, and bursty volatility. Multivariate with calendar features.
    Target is the first channel.
    """
    if num_samples <= 0:
        num_samples = 4096
    n_total = num_samples + seq_len + horizon + 200
    rng = np.random.default_rng(4)
    t = np.arange(n_total, dtype=np.float32)
    hour = t % 24
    weekday = ((t // 24) % 7).astype(np.float32)
    is_weekend = (weekday >= 5).astype(np.float32)
    # Two daily peaks: morning ~8 and evening ~19
    morning = np.exp(-((hour - 8) ** 2) / 8.0)
    evening = np.exp(-((hour - 19) ** 2) / 12.0)
    daily = 0.4 * morning + 0.6 * evening
    weekly = 1.0 - 0.2 * is_weekend
    annual_temp = 15 + 12 * np.sin(2 * np.pi * t / (24 * 365)) + rng.normal(scale=2.5, size=n_total)
    heat_cool = 0.04 * (annual_temp - 18) ** 2
    base = (1.0 + 0.6 * daily) * weekly + heat_cool / 5.0
    # AR component for autocorrelation, plus heteroscedastic shocks
    target = np.zeros(n_total, dtype=np.float32)
    target[0] = base[0]
    for i in range(1, n_total):
        vol = 0.05 + 0.04 * is_weekend[i]
        target[i] = 0.6 * target[i - 1] + 0.4 * base[i] + rng.normal(scale=vol)
    # Sin/cos hour-of-day, day-of-week features (good for TFT-like models)
    h_sin = np.sin(2 * np.pi * hour / 24)
    h_cos = np.cos(2 * np.pi * hour / 24)
    d_sin = np.sin(2 * np.pi * weekday / 7)
    d_cos = np.cos(2 * np.pi * weekday / 7)
    series = np.stack([target, annual_temp, h_sin, h_cos, d_sin, d_cos, is_weekend],
                      axis=1).astype(np.float32)
    return _make_windows(series, seq_len, horizon, num_samples)


def normalize_sequence(X_seq, y_seq=None):
    """Per-feature z-score on input windows; optional matching scale on target."""
    flat = X_seq.reshape(-1, X_seq.shape[-1])
    mean = flat.mean(axis=0)
    std = np.sqrt(((flat - mean) ** 2).mean(axis=0) + 1e-8)
    X_norm = (X_seq - mean) / std
    if y_seq is not None:
        # Target is the first channel
        y_norm = (y_seq - mean[0]) / std[0]
        return X_norm.astype(np.float32), y_norm.astype(np.float32), mean, std
    return X_norm.astype(np.float32), mean, std


def split_sequences(X, y, train_frac=0.8):
    """Chronological split (no shuffle) for sequence data."""
    split = int(len(X) * train_frac)
    return X[:split], y[:split], X[split:], y[split:]


def normalize_regression(X, y):
    """Z-score normalize features and centre/scale target."""
    X_mean = X.mean(axis=0)
    X_std = np.sqrt(((X - X_mean) ** 2).mean(axis=0) + 1e-8)
    y_mean = float(y.mean())
    y_std = float(np.sqrt(((y - y_mean) ** 2).mean() + 1e-8))
    return (X - X_mean) / X_std, (y - y_mean) / y_std, (X_mean, X_std, y_mean, y_std)


def shuffle_and_split_regression(X, y, train_frac=0.8):
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    split = int(len(X) * train_frac)
    return X[:split], y[:split], X[split:], y[split:]
