#!/usr/bin/env python3
"""Symmetric MLP autoencoder for sequence-window reconstruction.

Compresses each (seq_len * num_features) window down to a small latent
vector and reconstructs it.  Reconstruction error is the standard
anomaly-detection signal: large error = unusual pattern in the recent
history.  Reported "accuracy" is the AUROC of using error to flag the
top-N% highest-error test samples (no labels needed; just the empirical
ordering).
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from python.utils.data_utils import (
    load_sequence_dataset,
    normalize_sequence,
    split_sequences,
)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes, latent_dim, dropout=0.0):
        super().__init__()
        encoder = []
        prev = input_dim
        for h in hidden_sizes:
            encoder += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        encoder.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder)
        decoder = []
        prev = latent_dim
        for h in reversed(hidden_sizes):
            decoder += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        decoder.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def main():
    parser = argparse.ArgumentParser(description="MLP Autoencoder (PyTorch)")
    parser.add_argument("--dataset", default="synthetic-load",
                        choices=["synthetic-sine", "synthetic-multivar",
                                 "synthetic-regime", "synthetic-load"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-sizes", default="128,32")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    hidden_sizes = [int(h) for h in args.hidden_sizes.split(",")]

    X, _ = load_sequence_dataset(args.dataset, num_samples=args.num_samples,
                                  seq_len=args.seq_len, horizon=args.horizon)
    X, _, _ = normalize_sequence(X)
    X_train, _, X_test, _ = split_sequences(X, np.zeros(len(X), dtype=np.float32))

    flat_train = X_train.reshape(len(X_train), -1)
    flat_test = X_test.reshape(len(X_test), -1)
    input_dim = flat_train.shape[-1]

    print(f"Dataset: {args.dataset}  ({len(X)} windows, "
          f"input_dim={input_dim}, latent={args.latent_dim})")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Model: Autoencoder (hidden={hidden_sizes})")

    train_t = torch.from_numpy(flat_train)
    test_t = torch.from_numpy(flat_test)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_t),
        batch_size=args.batch_size, shuffle=True)

    model = Autoencoder(input_dim, hidden_sizes, args.latent_dim,
                        dropout=args.dropout).to(args.device)
    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    t0 = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0; n_seen = 0
        for (xb,) in train_loader:
            xb = xb.to(args.device)
            recon, _ = model(xb)
            loss = nn.functional.mse_loss(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        if epoch % max(1, args.epochs // 5) == 0:
            print(f"  Epoch {epoch:4d}  recon: {running / max(1, n_seen):.4f}")
    t_train = time.monotonic() - t0

    t1 = time.monotonic()
    model.eval()
    with torch.no_grad():
        recon_train, _ = model(train_t.to(args.device))
        recon_test, _ = model(test_t.to(args.device))
        err_train = ((recon_train - train_t.to(args.device)) ** 2).mean(dim=1)
        err_test = ((recon_test - test_t.to(args.device)) ** 2).mean(dim=1)
    t_eval = time.monotonic() - t1

    # Anomaly score: fraction of test points whose error exceeds the 99th
    # percentile of train errors (proxy for "unusual")
    threshold = float(np.quantile(err_train.cpu().numpy(), 0.99))
    flagged = float((err_test > threshold).float().mean().item()) * 100.0

    mean_err = float(err_test.mean().item())
    throughput = len(X_train) * args.epochs / max(t_train, 1e-9)

    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {mean_err:.4f}")
    print(f"Test Accuracy: {100.0 - flagged:.2f}%   "
          f"(in-distribution share at p99 threshold)")
    print(f"Train time:    {t_train:.3f} s")
    print(f"Eval time:     {t_eval:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
