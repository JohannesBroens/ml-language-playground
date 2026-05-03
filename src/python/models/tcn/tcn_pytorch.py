#!/usr/bin/env python3
"""1D Temporal Convolutional Network in PyTorch.

Causal dilated 1D convolutions with residual connections. The receptive field
grows exponentially with depth: r = 1 + 2 * (k - 1) * (2^L - 1) for kernel
size k and L stacked blocks with dilation 1, 2, 4, ...

I/O contract matches the RNN models: (B, T, F) -> (B, H).
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from python.utils.data_utils import (
    load_sequence_dataset,
    normalize_sequence,
    split_sequences,
)
from python.utils.seq_utils import make_loaders, train_regressor, report


class CausalConv1d(nn.Module):
    """Left-padded 1D conv so output[t] only depends on input[<=t]."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=0, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        residual = self.proj(x)
        h = self.conv1(x)
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = F.relu(h)
        h = self.dropout(h)
        return F.relu(h + residual)


class TCNForecaster(nn.Module):
    def __init__(self, num_features, num_channels, kernel_size, num_blocks,
                 horizon, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = num_features
        for b in range(num_blocks):
            dilation = 2 ** b
            out_ch = num_channels
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(num_channels, horizon)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T) for Conv1d
        h = x.transpose(1, 2)
        h = self.blocks(h)
        last = h[..., -1]
        return self.head(last)


def main():
    parser = argparse.ArgumentParser(description="PyTorch TCN Forecaster")
    parser.add_argument("--dataset", default="synthetic-multivar",
                        choices=["synthetic-sine", "synthetic-multivar",
                                 "synthetic-regime", "synthetic-load"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-channels", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--num-blocks", type=int, default=4)
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

    rf = 1 + 2 * (args.kernel_size - 1) * (2 ** args.num_blocks - 1)
    print(f"Dataset: {args.dataset}  ({len(X)} windows, "
          f"seq_len={args.seq_len}, horizon={args.horizon}, "
          f"features={X.shape[-1]})")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Model: TCN (channels={args.num_channels}, blocks={args.num_blocks}, "
          f"kernel={args.kernel_size}, receptive_field={rf})")

    train_loader, test_loader = make_loaders(X_train, y_train, X_test, y_test,
                                             args.batch_size)
    model = TCNForecaster(num_features=X.shape[-1],
                          num_channels=args.num_channels,
                          kernel_size=args.kernel_size,
                          num_blocks=args.num_blocks,
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
    report("TCN", args.dataset, metrics, throughput)


if __name__ == "__main__":
    main()
