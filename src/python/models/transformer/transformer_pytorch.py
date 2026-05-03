#!/usr/bin/env python3
"""Encoder-only Transformer forecaster in PyTorch.

Architecture:
    Linear input proj -> +sinusoidal positional encoding ->
    N x (multi-head self-attn + feed-forward, pre-norm residual) ->
    Mean-pool over time -> linear head -> horizon vector.

Mirrors the same I/O contract as the RNN/TCN models.
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from python.utils.data_utils import (
    load_sequence_dataset,
    normalize_sequence,
    split_sequences,
)
from python.utils.seq_utils import make_loaders, train_regressor, report


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, num_features, d_model, n_heads, num_layers,
                 dim_feedforward, horizon, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(num_features, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.norm(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Forecaster")
    parser.add_argument("--dataset", default="synthetic-multivar",
                        choices=["synthetic-sine", "synthetic-multivar",
                                 "synthetic-regime", "synthetic-load"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    X, y = load_sequence_dataset(args.dataset, num_samples=args.num_samples,
                                  seq_len=args.seq_len, horizon=args.horizon)
    X, y, _, _ = normalize_sequence(X, y)
    X_train, y_train, X_test, y_test = split_sequences(X, y)

    print(f"Dataset: {args.dataset}  ({len(X)} windows, "
          f"seq_len={args.seq_len}, horizon={args.horizon}, "
          f"features={X.shape[-1]})")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Model: Transformer (d_model={args.d_model}, heads={args.n_heads}, "
          f"layers={args.num_layers})")

    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test,
                                             args.batch_size)
    model = TransformerForecaster(num_features=X.shape[-1],
                                  d_model=args.d_model,
                                  n_heads=args.n_heads,
                                  num_layers=args.num_layers,
                                  dim_feedforward=args.dim_feedforward,
                                  horizon=args.horizon,
                                  dropout=args.dropout)
    metrics = train_regressor(model, train_loader, test_loader,
                              num_epochs=args.epochs,
                              lr=args.learning_rate,
                              optimizer=args.optimizer,
                              scheduler=args.scheduler,
                              device=args.device,
                              log_every=max(1, args.epochs // 5))
    throughput = len(X_train) * args.epochs / max(metrics["train_time"], 1e-9)
    report("Transformer", args.dataset, metrics, throughput)


if __name__ == "__main__":
    main()
