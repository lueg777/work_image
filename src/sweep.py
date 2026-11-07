"""Operator ablation driver: the actual experiment of this project.

Two modes, and they answer different questions. Be honest about which is which
when writing the report.

``--mode pretrained``
    Loads one checkpoint (normally the released ``NPR.pth``) and re-runs the
    cross-generator sweep with the input operator swapped underneath the frozen
    ``conv1`` weights. Cheap (no training), and it measures how much the learned
    filters *depend* on the exact residual they were trained on.

    Caveat that must appear in the report: this is a probe, not a retraining
    experiment. ``conv1`` was optimised for the nearest-neighbour residual, so a
    drop under Laplacian confounds "the operator is worse" with "these filters
    do not expect this input distribution". The clean claim needs ``scratch``.

``--mode scratch``
    Trains every operator from scratch with an identical recipe and compares
    final generalisation. This is the load-bearing experiment, and it costs
    roughly ``len(OPS) x (time of one training run)``.

    python -m src.sweep --mode pretrained --ckpt NPR.pth
    python -m src.sweep --mode scratch --ops npr_nearest,laplacian --epochs 50
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

from .build import REPO_ROOT, load_config
from .eval.cross_generator import run_cross_generator
from .eval.engine import load_checkpoint
from .models.npr_resnet import build_model
from .ops.residual_ops import OPERATOR_DESCRIPTIONS, REGISTRY
from .utils import ensure_dir, get_device, save_json, write_csv

ALL_OPS = list(REGISTRY)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="residual-operator ablation sweep")
    p.add_argument("--mode", choices=["pretrained", "scratch"],
                   default="pretrained")
    p.add_argument("--ops", default=",".join(ALL_OPS),
                   help="comma-separated operator names, or 'all'")
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", default="NPR.pth",
                   help="checkpoint for --mode pretrained")
    p.add_argument("--test-root", default=None)
    p.add_argument("--epochs", type=int, default=50,
                   help="epochs per operator for --mode scratch")
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap eval batches per generator (fast iteration)")
    p.add_argument("--out-dir", default="runs/sweep")
    p.add_argument("--workers", type=int, default=None,
                   help="dataloader workers; use 0 when multiprocessing is blocked")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def _resolve_ops(spec: str) -> list:
    if spec.strip().lower() == "all":
        return ALL_OPS
    ops = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [o for o in ops if o not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown operators {unknown}; available {ALL_OPS}")
    return ops


def run_pretrained(args, cfg, ops, out_dir) -> dict:
    device = get_device(args.device)
    test_root = args.test_root or cfg["data"]["test_root"]
    all_rows = []
    per_op = {}

    for op in ops:
        print(f"\n=== op={op}  ({OPERATOR_DESCRIPTIONS[op]}) ===")
        model = build_model({**cfg, "op": op}).to(device)
        rep = load_checkpoint(model, args.ckpt, device=device)
        if rep["missing"] or rep["unexpected"]:
            print(f"[warn] {op}: missing={len(rep['missing'])} "
                  f"unexpected={len(rep['unexpected'])}")
        res = run_cross_generator(model, test_root, cfg, device=device,
                                  op_name=op, max_batches=args.max_batches,
                                  progress=True)
        per_op[op] = res["summary"]
        for r in res["rows"]:
            all_rows.append(dict(r))
        s = res["summary"]
        print(f"  -> mean acc={s['mean_acc']*100:.2f} "
              f"auc={s['mean_auc']*100:.2f} ap={s['mean_ap']*100:.2f}")

    return _finish(all_rows, per_op, out_dir, "pretrained", ops)


def run_scratch(args, cfg, ops, out_dir) -> dict:
    all_rows = []
    per_op = {}
    for op in ops:
        run_dir = os.path.join(out_dir, "train", op)
        print(f"\n=== training op={op} -> {run_dir} ===")
        cmd = [sys.executable, "-m", "src.train", "--op", op,
               "--epochs", str(args.epochs), "--out", run_dir]
        if args.config:
            cmd += ["--config", args.config]
        if args.device:
            cmd += ["--device", args.device]
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

        ckpt = os.path.join(run_dir, "best.pth")
        device = get_device(args.device)
        model = build_model({**cfg, "op": op}).to(device)
        load_checkpoint(model, ckpt, device=device)
        test_root = args.test_root or cfg["data"]["test_root"]
        res = run_cross_generator(model, test_root, cfg, device=device,
                                  op_name=op, max_batches=args.max_batches,
                                  progress=True)
        per_op[op] = res["summary"]
        all_rows.extend(dict(r) for r in res["rows"])

    return _finish(all_rows, per_op, out_dir, "scratch", ops)


def _finish(all_rows, per_op, out_dir, mode, ops) -> dict:
    ensure_dir(out_dir)
    write_csv(all_rows, os.path.join(out_dir, "cross_all_ops.csv"))
    save_json({"mode": mode, "ops": ops, "summary": per_op},
              os.path.join(out_dir, "summary.json"))
    print_comparison(per_op, ops)
    print(f"\n[write] {out_dir}/cross_all_ops.csv and summary.json")
    return {"mode": mode, "ops": ops, "rows": all_rows, "summary": per_op}


def print_comparison(per_op, ops) -> None:
    print("\n== operator comparison (macro mean over generators) ==")
    w = max(len(o) for o in ops) + 2
    head = f"{'operator':<{w}}{'acc':>8}{'auc':>8}{'ap':>8}"
    print(head)
    print("-" * len(head))
    base = per_op.get("npr_nearest", {}).get("mean_acc")
    for op in ops:
        s = per_op.get(op, {})
        acc = s.get("mean_acc", float("nan"))
        if op == "npr_nearest":
            delta = "  (baseline)"
        elif base is not None and acc == acc and base == base:
            # percentage *points*, not percent: a 91.7 -> 84.9 drop is -6.8 pp
            delta = f"  ({100.0 * (acc - base):+.2f} pp vs npr_nearest)"
        else:
            delta = ""
        print(f"{op:<{w}}{acc*100:7.2f}{s.get('mean_auc', float('nan'))*100:8.2f}"
              f"{s.get('mean_ap', float('nan'))*100:8.2f}{delta}")


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.workers is not None:
        cfg.setdefault("data", {})["workers"] = args.workers
    ops = _resolve_ops(args.ops)
    out_dir = ensure_dir(args.out_dir)
    print(f"mode={args.mode}  ops={ops}")
    if args.mode == "pretrained":
        run_pretrained(args, cfg, ops, out_dir)
    else:
        run_scratch(args, cfg, ops, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
