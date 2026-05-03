#!/usr/bin/env python3
"""Temporal Fusion Transformer (Lim et al., 2021).

Faithful PyTorch implementation of the major building blocks:

  - Gated Linear Unit (GLU)
  - Gated Residual Network (GRN)
  - Variable Selection Network (VSN) for past, known-future, and static inputs
  - Static covariate encoders producing four context vectors
  - LSTM seq2seq layer (encoder over past, decoder over known future) for
    locality-aware processing
  - Static enrichment GRN
  - Interpretable multi-head attention (shared value head, per-head queries/keys)
  - Position-wise feed-forward GRN
  - Quantile output head producing (B, H, Q) — one prediction per horizon
    step per quantile

The synthetic datasets here don't have explicit static / known-future
fields, so we derive them automatically:

    past inputs       : full feature window for the lookback horizon
    known future      : all non-target features (assumed forecast-able)
    static covariates : trailing-window mean and std of the target

This gives the model the same structural inputs it sees on real
forecasting problems, just constructed from generic synthetic data.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from python.utils.data_utils import (
    load_sequence_dataset,
    normalize_sequence,
    split_sequences,
)


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------

class GLU(nn.Module):
    """Gated Linear Unit: (W x) * sigmoid(V x).

    Allows the network to learn how much of an upstream signal to let
    through, similar in spirit to the gates inside an LSTM.
    """
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(x) * torch.sigmoid(self.gate(x))


class GRN(nn.Module):
    """Gated Residual Network.

        eta_2 = ELU(W1 x + W2 c + b1)
        eta_1 = W3 eta_2 + b2
        out   = LayerNorm(skip + GLU(Dropout(eta_1)))

    `c` is an optional static context vector (broadcast across the time axis
    if `x` is sequential).  When the input dim differs from the output dim we
    project the skip connection to the right shape.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1,
                 context_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_context = nn.Linear(context_dim, hidden_dim, bias=False) \
            if context_dim is not None else None
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) \
            if input_dim != output_dim else nn.Identity()

    def forward(self, x, context=None):
        residual = self.skip(x)
        h = self.fc1(x)
        if context is not None and self.fc_context is not None:
            ctx = self.fc_context(context)
            if ctx.dim() == 2 and h.dim() == 3:
                ctx = ctx.unsqueeze(1)
            h = h + ctx
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)
        return self.norm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """Variable selection over a set of per-feature embeddings.

    Each input feature gets its own GRN that maps its raw scalar to a
    `hidden_dim`-dimensional embedding.  A second GRN over the flattened
    concatenation of those embeddings (optionally conditioned on a static
    context) produces a softmax weight per variable.  The output is the
    weighted sum of the embeddings.

    This gives the model both interpretability (the softmax weights) and a
    learned form of regularisation (irrelevant variables get downweighted).
    """
    def __init__(self, num_inputs, hidden_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        self.feature_grns = nn.ModuleList([
            GRN(1, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_inputs)
        ])
        self.weight_grn = GRN(num_inputs * hidden_dim, hidden_dim, num_inputs,
                              dropout=dropout, context_dim=context_dim)

    def forward(self, x, context=None):
        # x: (B, T, num_inputs) or (B, num_inputs)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_time = True
        else:
            squeeze_time = False
        B, T, F = x.shape
        embedded = []
        for i in range(F):
            xi = x[..., i:i + 1]                 # (B, T, 1)
            embedded.append(self.feature_grns[i](xi))    # (B, T, hidden)
        stacked = torch.stack(embedded, dim=-2)  # (B, T, F, hidden)
        flat = stacked.flatten(start_dim=-2)     # (B, T, F*hidden)
        weights = self.weight_grn(flat, context=context)  # (B, T, F)
        weights = F_softmax(weights, dim=-1).unsqueeze(-1)  # (B, T, F, 1)
        combined = (stacked * weights).sum(dim=-2)        # (B, T, hidden)
        if squeeze_time:
            combined = combined.squeeze(1)
            weights = weights.squeeze(1)
        return combined, weights.squeeze(-1)


def F_softmax(x, dim):
    """Stable softmax with float32 fallback for half-precision."""
    return F.softmax(x, dim=dim)


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with a shared value projection across heads.

    The standard multi-head attention learns separate W_v per head, which
    obscures which timestep contributed to a prediction.  TFT shares the
    value projection so that head-averaged attention weights have a clean
    interpretation as importance per timestep.
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.ModuleList([nn.Linear(d_model, self.d_head, bias=False)
                                     for _ in range(n_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, self.d_head, bias=False)
                                     for _ in range(n_heads)])
        self.v_proj = nn.Linear(d_model, self.d_head, bias=False)
        self.out = nn.Linear(self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        head_outputs = []
        head_attns = []
        v_shared = self.v_proj(v)
        scale = math.sqrt(self.d_head)
        for h in range(self.n_heads):
            qh = self.q_proj[h](q)
            kh = self.k_proj[h](k)
            scores = qh @ kh.transpose(-2, -1) / scale  # (B, Tq, Tk)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            head_outputs.append(attn @ v_shared)        # (B, Tq, d_head)
            head_attns.append(attn)
        head_avg = torch.stack(head_outputs, dim=0).mean(dim=0)  # (B, Tq, d_head)
        attn_avg = torch.stack(head_attns, dim=0).mean(dim=0)
        out = self.out(head_avg)
        return out, attn_avg


# ---------------------------------------------------------------------------
#  Full model
# ---------------------------------------------------------------------------

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_past_features, num_future_features, num_static_features,
                 hidden_size, n_heads, horizon, num_quantiles=3, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.num_quantiles = num_quantiles

        # Static encoder: a VSN over scalar static features
        self.static_vsn = VariableSelectionNetwork(num_static_features,
                                                   hidden_size, dropout=dropout)

        # Static context encoders — four separate GRNs as in the paper
        self.static_ctx_vs = GRN(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.static_ctx_enr = GRN(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.static_ctx_h = GRN(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.static_ctx_c = GRN(hidden_size, hidden_size, hidden_size, dropout=dropout)

        # Past / future variable selection (conditioned on static context)
        self.past_vsn = VariableSelectionNetwork(num_past_features, hidden_size,
                                                 dropout=dropout, context_dim=hidden_size)
        self.future_vsn = VariableSelectionNetwork(num_future_features, hidden_size,
                                                   dropout=dropout, context_dim=hidden_size)

        # LSTM encoder/decoder for locality
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm_glu = GLU(hidden_size)
        self.lstm_norm = nn.LayerNorm(hidden_size)

        # Static enrichment
        self.static_enrichment = GRN(hidden_size, hidden_size, hidden_size,
                                     dropout=dropout, context_dim=hidden_size)

        # Self-attention block
        self.attn = InterpretableMultiHeadAttention(hidden_size, n_heads, dropout=dropout)
        self.attn_glu = GLU(hidden_size)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Position-wise feed-forward
        self.pos_ff = GRN(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.pos_glu = GLU(hidden_size)
        self.pos_norm = nn.LayerNorm(hidden_size)

        # Output projection: per-step quantile predictions
        self.output = nn.Linear(hidden_size, num_quantiles)

    def forward(self, past, future, static):
        """
        past:   (B, T_past, num_past_features)
        future: (B, T_future, num_future_features)   T_future == horizon
        static: (B, num_static_features)
        Returns: (B, horizon, num_quantiles)
        """
        # 1. Static encoding
        static_vec, _ = self.static_vsn(static)
        c_vs = self.static_ctx_vs(static_vec)
        c_enr = self.static_ctx_enr(static_vec)
        c_h = self.static_ctx_h(static_vec).unsqueeze(0)  # (1, B, H)
        c_c = self.static_ctx_c(static_vec).unsqueeze(0)

        # 2. Variable selection on past and future, conditioned on c_vs
        past_emb, _ = self.past_vsn(past, context=c_vs)
        future_emb, _ = self.future_vsn(future, context=c_vs)

        # 3. LSTM seq2seq with static-conditioned init state
        enc_out, (h_n, c_n) = self.encoder_lstm(past_emb, (c_h, c_c))
        dec_out, _ = self.decoder_lstm(future_emb, (h_n, c_n))
        full = torch.cat([enc_out, dec_out], dim=1)
        skip = torch.cat([past_emb, future_emb], dim=1)
        gated = self.lstm_glu(full)
        full = self.lstm_norm(skip + gated)

        # 4. Static enrichment
        enriched = self.static_enrichment(full, context=c_enr)

        # 5. Causal multi-head attention over the temporal axis
        T = enriched.size(1)
        # Lower-triangular mask: position i can only attend to j <= i
        mask = torch.tril(torch.ones(T, T, device=enriched.device,
                                     dtype=enriched.dtype))
        attn_out, attn_w = self.attn(enriched, enriched, enriched, attn_mask=mask)
        attn_out = self.attn_glu(attn_out)
        attn_out = self.attn_norm(enriched + attn_out)

        # 6. Position-wise feed-forward
        ff = self.pos_ff(attn_out)
        ff = self.pos_glu(ff)
        ff = self.pos_norm(attn_out + ff)

        # 7. Take the decoder portion and project to quantiles
        decoder_out = ff[:, -self.horizon:, :]
        return self.output(decoder_out)


# ---------------------------------------------------------------------------
#  Data plumbing — derive past/future/static views from a generic windowed
#  multivariate dataset
# ---------------------------------------------------------------------------

def derive_tft_views(X_seq, y_seq, horizon, target_idx=0):
    """Split each (B, T, F) window into:

        past   : whole window of features
        future : non-target features for the horizon (assumed forecast-able)
        static : (target_mean, target_std) over the past window

    For these synthetic datasets we don't have a separate "future" segment,
    so we approximate by repeating the last `horizon` future-feature rows;
    in real deployments these would be replaced with actual forecasts.
    """
    past = X_seq.copy()
    last_future = X_seq[:, -horizon:, :]
    mask = np.ones(X_seq.shape[-1], dtype=bool)
    mask[target_idx] = False
    future = last_future[:, :, mask]
    target_window = X_seq[..., target_idx]
    static = np.stack([target_window.mean(axis=1),
                       target_window.std(axis=1)], axis=1).astype(np.float32)
    return past.astype(np.float32), future.astype(np.float32), static.astype(np.float32)


class TFTDataset(torch.utils.data.Dataset):
    def __init__(self, past, future, static, y):
        self.past = torch.from_numpy(past)
        self.future = torch.from_numpy(future)
        self.static = torch.from_numpy(static)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.past[i], self.future[i], self.static[i], self.y[i]


def cosine_lr(epoch, total, lr_max, warmup, lr_min=1e-6):
    if epoch < warmup:
        return lr_max * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def train_tft(model, train_loader, test_loader, taus, num_epochs, lr,
              optimizer, scheduler, device, log_every):
    model = model.to(device)
    taus_t = torch.tensor(taus, device=device, dtype=torch.float32)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    warmup = max(1, int(num_epochs * 0.05)) if scheduler == "cosine" else 0

    t0 = time.monotonic()
    for epoch in range(num_epochs):
        if scheduler == "cosine":
            for g in opt.param_groups:
                g["lr"] = cosine_lr(epoch, num_epochs, lr, warmup)
        model.train()
        running = 0.0
        n_seen = 0
        for past, future, static, y in train_loader:
            past = past.to(device); future = future.to(device)
            static = static.to(device); y = y.to(device)
            pred = model(past, future, static)         # (B, H, Q)
            target = y.unsqueeze(-1)                   # (B, H, 1)
            resid = target - pred
            loss = torch.maximum(taus_t * resid,
                                 (taus_t - 1.0) * resid).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * past.size(0)
            n_seen += past.size(0)
        if log_every and epoch % log_every == 0:
            print(f"  Epoch {epoch:4d}  pinball: {running / max(1, n_seen):.4f}")
    t_train = time.monotonic() - t0

    # Eval
    model.eval()
    t1 = time.monotonic()
    total_p, n_p = 0.0, 0
    in_band = 0
    sse_med, sst_med, ny_med = 0.0, 0.0, 0
    y_mean = 0.0
    n_y = 0
    with torch.no_grad():
        for _, _, _, y in test_loader:
            n_y += y.numel()
            y_mean += float(y.sum().item())
        if n_y > 0:
            y_mean /= n_y
        i_lo = taus.index(min(taus))
        i_hi = taus.index(max(taus))
        try:
            i_med = taus.index(0.5)
        except ValueError:
            i_med = len(taus) // 2
        for past, future, static, y in test_loader:
            past = past.to(device); future = future.to(device)
            static = static.to(device); y = y.to(device)
            pred = model(past, future, static)
            target = y.unsqueeze(-1)
            resid = target - pred
            pinball = torch.maximum(taus_t * resid,
                                    (taus_t - 1.0) * resid)
            total_p += float(pinball.sum().item())
            n_p += pinball.numel()
            lo = pred[..., i_lo]
            hi = pred[..., i_hi]
            in_band += int(((y >= lo) & (y <= hi)).sum().item())
            med = pred[..., i_med]
            err = med - y
            sse_med += float((err ** 2).sum().item())
            sst_med += float(((y - y_mean) ** 2).sum().item())
            ny_med += y.numel()
    t_eval = time.monotonic() - t1
    return {
        "pinball": total_p / max(1, n_p),
        "coverage": in_band / max(1, ny_med),
        "rmse_median": math.sqrt(sse_med / max(1, ny_med)),
        "r2_median": 1.0 - sse_med / sst_med if sst_med > 0 else 0.0,
        "train_time": t_train,
        "eval_time": t_eval,
    }


def main():
    parser = argparse.ArgumentParser(description="Temporal Fusion Transformer (PyTorch)")
    parser.add_argument("--dataset", default="synthetic-load",
                        choices=["synthetic-sine", "synthetic-multivar",
                                 "synthetic-regime", "synthetic-load"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", default="cosine", choices=["none", "cosine"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    taus = [float(x) for x in args.quantiles.split(",")]

    X, y = load_sequence_dataset(args.dataset, num_samples=args.num_samples,
                                  seq_len=args.seq_len, horizon=args.horizon)
    X, y, _, _ = normalize_sequence(X, y)
    X_train, y_train, X_test, y_test = split_sequences(X, y)
    past_tr, fut_tr, st_tr = derive_tft_views(X_train, y_train, args.horizon)
    past_te, fut_te, st_te = derive_tft_views(X_test, y_test, args.horizon)

    print(f"Dataset: {args.dataset}  ({len(X)} windows, "
          f"seq_len={args.seq_len}, horizon={args.horizon}, "
          f"features={X.shape[-1]})")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Model: TFT (hidden={args.hidden_size}, heads={args.n_heads}, "
          f"quantiles={taus})")
    print(f"  past_features={past_tr.shape[-1]}  "
          f"future_features={fut_tr.shape[-1]}  "
          f"static_features={st_tr.shape[-1]}")

    train_loader = torch.utils.data.DataLoader(
        TFTDataset(past_tr, fut_tr, st_tr, y_train),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        TFTDataset(past_te, fut_te, st_te, y_test),
        batch_size=args.batch_size, shuffle=False)

    model = TemporalFusionTransformer(
        num_past_features=past_tr.shape[-1],
        num_future_features=fut_tr.shape[-1],
        num_static_features=st_tr.shape[-1],
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        horizon=args.horizon,
        num_quantiles=len(taus),
        dropout=args.dropout,
    )
    metrics = train_tft(model, train_loader, test_loader, taus,
                        args.epochs, args.learning_rate,
                        args.optimizer, args.scheduler, args.device,
                        log_every=max(1, args.epochs // 5))
    throughput = len(X_train) * args.epochs / max(metrics["train_time"], 1e-9)

    print(f"\n=== Results on {args.dataset} ===")
    print(f"Test Loss:     {metrics['pinball']:.4f}")
    print(f"Test Accuracy: {metrics['coverage'] * 100:.2f}%")
    print(f"Median RMSE:   {metrics['rmse_median']:.4f}")
    print(f"Median R^2:    {metrics['r2_median']:.4f}")
    print(f"Train time:    {metrics['train_time']:.3f} s")
    print(f"Eval time:     {metrics['eval_time']:.3f} s")
    print(f"Throughput:    {throughput:.0f} samples/s")


if __name__ == "__main__":
    main()
