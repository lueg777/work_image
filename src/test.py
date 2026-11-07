"""Evaluate a checkpoint on unseen generators, with and without attacks.

    # zero-shot with the released NPR weights, nearest operator only
    python -m src.test --ckpt NPR.pth --op npr_nearest --cross --robustness

    # a model you trained yourself
    python -m src.test --ckpt runs/laplacian/best.pth --cross --out-json out.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from .build import load_config
from .eval.cross_generator import format_table, run_cross_generator
from .eval.engine import load_checkpoint
from .eval.robustness import format_matrix, run_robustness
from .models.npr_resnet import build_model, count_parameters
from .utils import ensure_dir, get_device, save_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="evaluate NPR-family detector")
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt", required=True, help="path to a .pth checkpoint")
    p.add_argument("--op", default=None,
                   help="override the residual operator (default: from config)")
    p.add_argument("--test-root", default=None,
                   help="root containing <generator>/{real,fake}/ folders")
    p.add_argument("--cross", action="store_true", help="run cross-generator sweep")
    p.add_argument("--robustness", action="store_true", help="run attack sweep")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--limit-per-class", type=int, default=None,
                   help="cap each class at N images, balanced (use this instead of "
                        "--max-batches: truncating in file order leaves a "
                        "single-class sample and AUC/AP come back NaN)")
    p.add_argument("--workers", type=int, default=None,
                   help="dataloader workers; use 0 when multiprocessing is blocked")
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-dir", default=None, help="where to write per-run CSVs")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.workers is not None:
        cfg.setdefault("data", {})["workers"] = args.workers
    if args.limit_per_class:
        cfg.setdefault("data", {})["eval_limit_per_class"] = args.limit_per_class
    op = args.op or cfg.get("op", "npr_nearest")
    test_root = args.test_root or cfg["data"]["test_root"]
    if args.config is None and args.test_root:
        print("[warn] --test-root given without --config: crop_size / load_size / "
              "normalisation still come from configs/default.yaml "
              f"(crop={cfg['data'].get('crop_size')}, "
              f"resize={cfg['data'].get('load_size')}). Evaluating a small "
              "dataset upscaled to 256px yields meaningless numbers -- pass the "
              "same --config you trained with.")
    device = get_device(args.device)

    model = build_model({**cfg, "op": op}).to(device)
    print(f"op={op}  params={count_parameters(model):,}  device={device}")
    report = load_checkpoint(model, args.ckpt, device=device)
    print(f"[ckpt] loaded {report['n_loaded']}/{report['n_in_checkpoint']} tensors "
          f"from {report['path']}")
    if report["missing"]:
        print(f"[ckpt] MISSING ({len(report['missing'])}): "
              f"{report['missing'][:6]}{' ...' if len(report['missing']) > 6 else ''}")
    if report["unexpected"]:
        print(f"[ckpt] UNEXPECTED ({len(report['unexpected'])}): "
              f"{report['unexpected'][:6]}")

    result = {"op": op, "checkpoint": report, "cfg_test_root": test_root}
    out_dir = ensure_dir(args.out_dir) if args.out_dir else None

    if args.cross or not args.robustness:
        print("\n== cross-generator ==")
        csv = os.path.join(out_dir, f"cross_{op}.csv") if out_dir else None
        result["cross"] = run_cross_generator(
            model, test_root, cfg, device=device, op_name=op,
            max_batches=args.max_batches, out_csv=csv)
        print(format_table(result["cross"]["rows"], "acc"))
        s = result["cross"]["summary"]
        print(f"mean acc={s['mean_acc']*100:.2f}  auc={s['mean_auc']*100:.2f}  "
              f"ap={s['mean_ap']*100:.2f}")

    if args.robustness:
        print("\n== robustness ==")
        csv = os.path.join(out_dir, f"robust_{op}.csv") if out_dir else None
        result["robustness"] = run_robustness(
            model, test_root, cfg, device=device, op_name=op,
            max_batches=args.max_batches, out_csv=csv)
        print(format_matrix(result["robustness"]["rows"], "acc"))

    if args.out_json:
        save_json(result, args.out_json)
        print(f"\n[write] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
