#!/usr/bin/env python3
"""LSTM sequence-to-horizon model in PyTorch.

Input:  (batch, seq_len, num_features)
Output: (batch, horizon)

Uses an encoder-only LSTM, takes the final hidden state, and projects to a
flat horizon vector. Simple and fast; a stronger variant would do recursive
or attention-based decoding, which is what the Transformer/TFT models do.
"""

import argparse
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
from python.utils.seq_utils import make_loaders, train_regressor, report


class LSTMForecaster(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers, horizon, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def main():
    parser = argparse.ArgumentParser(description="PyTorch LSTM Forecaster")
    parser.add_argument("--dataset", default="synthetic-multivar",
                        choices=["synthetic-sine", "synthetic-multivar",
                                 "synthetic-regime", "synthetic-load"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
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
    print(f"Model: LSTM (hidden={args.hidden_size}, layers={args.num_layers})")

    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test,
                                             args.batch_size)
    model = LSTMForecaster(num_features=X.shape[-1],
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
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
    report("LSTM", args.dataset, metrics, throughput)


if __name__ == "__main__":
    main()
