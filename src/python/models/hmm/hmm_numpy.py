#!/usr/bin/env python3
"""Gaussian Hidden Markov Model with Baum-Welch (EM) training and Viterbi decoding.

  - Hidden states are discrete (K).
  - Emission per state is a univariate Gaussian (mu_k, sigma_k).
  - Transitions are a K x K stochastic matrix.

Trained on a single long sequence reconstructed from windowed data;
reported metric is regime-classification accuracy when ground truth states
are available, plus the log-likelihood per observation.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from python.utils.data_utils import load_sequence_dataset


def gaussian_logpdf(x, mu, sigma):
    var = sigma ** 2 + 1e-9
    return -0.5 * (math.log(2 * math.pi) + np.log(var) + (x - mu) ** 2 / var)


def forward_log(log_emit, log_pi, log_T):
    """Numerically stable forward pass in log-space."""
    n, K = log_emit.shape
    log_alpha = np.empty_like(log_emit)
    log_alpha[0] = log_pi + log_emit[0]
    for t in range(1, n):
        # log_alpha[t, k] = log sum_j alpha[t-1, j] T[j, k] + emit[t, k]
        m = log_alpha[t - 1].max()
        # vectorised log-sum-exp
        log_alpha[t] = (m + np.log(np.exp(log_alpha[t - 1] - m) @ np.exp(log_T))) + log_emit[t]
    return log_alpha


def backward_log(log_emit, log_T):
    n, K = log_emit.shape
    log_beta = np.zeros_like(log_emit)
    for t in range(n - 2, -1, -1):
        m = (log_emit[t + 1] + log_beta[t + 1]).max()
        log_beta[t] = m + np.log(np.exp(log_T) @ np.exp(log_emit[t + 1] + log_beta[t + 1] - m))
    return log_beta


def viterbi(log_emit, log_pi, log_T):
    n, K = log_emit.shape
    delta = np.empty_like(log_emit)
    psi = np.empty((n, K), dtype=np.int32)
    delta[0] = log_pi + log_emit[0]
    psi[0] = -1
    for t in range(1, n):
        scores = delta[t - 1][:, None] + log_T            # (K_prev, K_next)
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + log_emit[t]
    path = np.empty(n, dtype=np.int32)
    path[-1] = int(delta[-1].argmax())
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, float(delta[-1].max())


def baum_welch(x, K, num_iter, rng):
    n = len(x)
    pi = rng.dirichlet(np.ones(K))
    T = rng.dirichlet(np.ones(K), size=K)
    mu = rng.choice(x, size=K, replace=False).astype(np.float64)
    sigma = np.full(K, x.std() + 1e-3, dtype=np.float64)

    for it in range(num_iter):
        log_emit = np.stack([gaussian_logpdf(x, mu[k], sigma[k]) for k in range(K)], axis=1)
        log_pi = np.log(pi + 1e-12)
        log_T = np.log(T + 1e-12)
        log_alpha = forward_log(log_emit, log_pi, log_T)
        log_beta = backward_log(log_emit, log_T)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - log_gamma.max(axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        # xi[t, i, j] proportional to alpha[t, i] T[i, j] emit[t+1, j] beta[t+1, j]
        # Sum over t to update T
        xi_sum = np.zeros((K, K))
        for t in range(n - 1):
            num = (np.exp(log_alpha[t])[:, None]
                   * T
                   * np.exp(log_emit[t + 1] + log_beta[t + 1])[None, :])
            num = num / max(num.sum(), 1e-300)
            xi_sum += num

        # M-step
        pi = gamma[0]
        T = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
        weights = gamma.sum(axis=0)
        mu = (gamma * x[:, None]).sum(axis=0) / np.maximum(weights, 1e-12)
        var = (gamma * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / np.maximum(weights, 1e-12)
        sigma = np.sqrt(np.maximum(var, 1e-6))
    return pi, T, mu, sigma


def main():
    parser = argparse.ArgumentParser(description="Gaussian HMM (NumPy)")
    parser.add_argument("--dataset", default="synthetic-regime",
                        choices=["synthetic-sine", "synthetic-multivar",
                                 "synthetic-regime", "synthetic-load"])
    parser.add_argument("--num-states", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    X_seq, _ = load_sequence_dataset(args.dataset, num_samples=args.num_samples,
                                      seq_len=args.seq_len, horizon=args.horizon)
    # Use the first channel of the first window to reconstruct a single long
    # sequence in chronological order.
    series = X_seq[:, :, 0].reshape(-1).astype(np.float64)
    series = (series - series.mean()) / (series.std() + 1e-8)

    print(f"Dataset: {args.dataset}  (series length={len(series)})")
    print(f"Model: HMM (K={args.num_states} states, gaussian emissions)")

    t0 = time.monotonic()
    pi, T, mu, sigma = baum_welch(series, args.num_states, args.epochs,
                                  rng=np.random.default_rng(7))
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    log_emit = np.stack([gaussian_logpdf(series, mu[k], sigma[k])
                         for k in range(args.num_states)], axis=1)
    path, ll = viterbi(log_emit, np.log(pi + 1e-12), np.log(T + 1e-12))
    log_alpha = forward_log(log_emit, np.log(pi + 1e-12), np.log(T + 1e-12))
    # Total log likelihood = log sum exp alpha[T-1]
    m = log_alpha[-1].max()
    total_ll = m + math.log(float(np.exp(log_alpha[-1] - m).sum()))
    avg_ll = total_ll / len(series)
    t_eval = time.monotonic() - t1

    state_counts = np.bincount(path, minlength=args.num_states)
    throughput = len(series) * args.epochs / max(t_train, 1e-9)

    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {-avg_ll:.4f}")          # neg log-lik per obs
    print(f"Test Accuracy: {100.0 * state_counts.max() / len(series):.2f}%  "
          f"(majority-state share)")
    print(f"State means:   {mu.tolist()}")
    print(f"State stds:    {sigma.tolist()}")
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
