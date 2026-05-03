#!/usr/bin/env python3
"""Benchmark driver for the regression / sequence model families.

The classification benchmark in benchmark.py is tightly coupled to MLP/CNN
metrics and the matrix-multiply-heavy hyperparameter axes that are
meaningless for trees, GPs, kernel methods, and seq2seq forecasters.  This
script reads `configs/models/regression.yaml` and `configs/models/sequence.yaml`,
runs every implementation it finds, parses the standardised output block,
and prints a markdown comparison table.

Usage:
    python src/scripts/extras_benchmark.py --family regression
    python src/scripts/extras_benchmark.py --family sequence
    python src/scripts/extras_benchmark.py --family all
"""

import argparse
import os
import re
import subprocess
import sys
import time

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs", "models")

PATTERN_LOSS = re.compile(r"Test Loss:\s+([\d.]+)")
PATTERN_ACC = re.compile(r"Test Accuracy:\s+([\d.]+)%")
PATTERN_TRAIN = re.compile(r"Train time:\s+([\d.]+)\s*s")
PATTERN_EVAL = re.compile(r"Eval time:\s+([\d.]+)\s*s")
PATTERN_THROUGHPUT = re.compile(r"Throughput:\s+([\d.]+)\s*samples/s")


def _build_cmd(template, dataset, overrides=None):
    cmd = (template
           .replace("{python}", sys.executable)
           .replace("{dataset}", dataset))
    if overrides:
        for k, v in overrides.items():
            # Only append flag if not already present in the template
            if f"--{k}" not in cmd:
                cmd = cmd + f" --{k} {v}"
    return cmd


def _ld_env(label):
    env = os.environ.copy()
    if label.startswith("Regression C"):
        build_dir = os.path.join(PROJECT_ROOT, "src", "c", "build_cpu")
        lib_dirs = [build_dir,
                    os.path.join(build_dir, "models", "regression")]
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def _is_available(label):
    binary_map = {
        "Regression C (CPU)":   "src/c/build_cpu/regression_main",
        "Regression Rust (CPU)": "src/rust/target/release/regression-cpu",
    }
    if label in binary_map:
        return os.path.isfile(os.path.join(PROJECT_ROOT, binary_map[label]))
    if "PyTorch" in label or "NumPy" in label:
        # Pure-python implementations are always callable
        return True
    return True


def run_one(label, cmd, timeout=600):
    env = _ld_env(label)
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True,
                                 timeout=timeout, cwd=PROJECT_ROOT, env=env)
        out = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        sys.stderr.write(f"FAIL  {label}: returncode={result.returncode}\n")
        sys.stderr.write(out[:500] + "\n")
        return None
    m_loss = PATTERN_LOSS.search(out)
    m_acc = PATTERN_ACC.search(out)
    m_tr = PATTERN_TRAIN.search(out)
    m_ev = PATTERN_EVAL.search(out)
    m_tp = PATTERN_THROUGHPUT.search(out)
    if not all([m_loss, m_acc, m_tr, m_ev]):
        sys.stderr.write(f"PARSE-FAIL {label}\n")
        return None
    parsed = {
        "loss": float(m_loss.group(1)),
        "acc": float(m_acc.group(1)),
        "train": float(m_tr.group(1)),
        "eval": float(m_ev.group(1)),
    }
    if m_tp:
        parsed["throughput"] = float(m_tp.group(1))
    return parsed


def run_family(family_path, overrides=None, dataset_filter=None,
               impl_filter=None, save_json=None):
    with open(family_path) as f:
        cfg = yaml.safe_load(f)
    bench = cfg.get("benchmark", {})
    datasets = bench.get("datasets", [])
    if dataset_filter:
        datasets = [d for d in datasets if d in dataset_filter]
    impls = cfg.get("implementations", [])
    if impl_filter:
        impls = [im for im in impls
                 if any(f.lower() in im["name"].lower() for f in impl_filter)]
    timeout = cfg.get("scaling", {}).get("timeout", 600)
    results = {}                # results[dataset][label] = parsed
    for ds in datasets:
        results[ds] = {}
        for impl in impls:
            label = impl["name"]
            if not _is_available(label):
                continue
            ds_eff = impl.get("dataset_override") or ds
            cmd = _build_cmd(impl["cmd"], ds_eff, overrides=overrides)
            t0 = time.monotonic()
            parsed = run_one(label, cmd, timeout=timeout)
            wall = time.monotonic() - t0
            label_disp = label
            if impl.get("dataset_override"):
                label_disp = f"{label} [{ds_eff}]"
            print(f"  [{ds:>22s}] {label_disp:<35s}  "
                  f"{wall:6.2f}s  "
                  + (f"loss={parsed['loss']:.3f}  acc={parsed['acc']:.2f}%"
                     if parsed else "FAILED"))
            if parsed:
                parsed["wall_total"] = wall
                results[ds][label_disp] = parsed
    if save_json:
        import json
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
    return results


def print_markdown(results, family_name):
    print(f"\n## {family_name.title()} family — benchmark results\n")
    print("| Dataset | Implementation | Loss | Score (%) | Train (s) | Eval (s) |")
    print("|---|---|---|---|---|---|")
    for ds in sorted(results):
        for label, p in sorted(results[ds].items()):
            print(f"| {ds} | {label} | {p['loss']:.4f} | {p['acc']:.2f} "
                  f"| {p['train']:.3f} | {p['eval']:.3f} |")


def main():
    parser = argparse.ArgumentParser(description="Benchmark driver for regression/sequence model families")
    parser.add_argument("--family", default="regression",
                        choices=["regression", "sequence", "all"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs across all impls")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Override --num-samples on all impls")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated subset of datasets")
    parser.add_argument("--impls", type=str, default=None,
                        help="Comma-separated substring filter on impl names")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Path to save raw results as JSON (default: results/cache/extras_<family>.json)")
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None: overrides["epochs"] = args.epochs
    if args.num_samples is not None: overrides["num-samples"] = args.num_samples
    ds_filter = args.datasets.split(",") if args.datasets else None
    impl_filter = args.impls.split(",") if args.impls else None

    if args.family in ("regression", "all"):
        path = os.path.join(CONFIGS_DIR, "regression.yaml")
        print("=" * 70)
        print("  Regression family")
        print("=" * 70)
        save = args.save_json or os.path.join(PROJECT_ROOT, "results", "cache", "extras_regression.json")
        results = run_family(path, overrides=overrides,
                              dataset_filter=ds_filter, impl_filter=impl_filter,
                              save_json=save)
        print_markdown(results, "regression")
    if args.family in ("sequence", "all"):
        path = os.path.join(CONFIGS_DIR, "sequence.yaml")
        print("\n" + "=" * 70)
        print("  Sequence family")
        print("=" * 70)
        save = args.save_json or os.path.join(PROJECT_ROOT, "results", "cache", "extras_sequence.json")
        if args.family == "all":
            save = os.path.join(PROJECT_ROOT, "results", "cache", "extras_sequence.json")
        results = run_family(path, overrides=overrides,
                              dataset_filter=ds_filter, impl_filter=impl_filter,
                              save_json=save)
        print_markdown(results, "sequence")


if __name__ == "__main__":
    main()
