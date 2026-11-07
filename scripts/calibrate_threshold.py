"""Is the accuracy gap a threshold problem or a ranking problem?

The report claims "AUC is fine, the 9 pp loss is a threshold offset". That claim
is only worth writing down if it is measured, so this script measures it:

  * collect raw logits per generator,
  * split each class in half (tune / evaluate, fixed seed so it is reproducible),
  * pick the threshold that maximises accuracy on the tuning half,
  * report accuracy on the held-out half at that threshold, next to the
    threshold-0 accuracy and the oracle accuracy on the same half.

``acc_best`` alone would be an oracle (threshold chosen with labels in hand) and
would overstate what calibration can buy, which is exactly the mistake the report
should not make.

Usage:
    python scripts/calibrate_threshold.py --config configs/eval_256_nocrop.yaml \
        --ckpt ../database/weights/NPR.pth --limit-per-class 300 \
        --out results/calibration.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.build import build_eval_loader, load_config  # noqa: E402
from src.eval.cross_generator import _family  # noqa: E402
from src.eval.engine import collect_scores, load_checkpoint  # noqa: E402
from src.eval.metrics import accuracy, auc  # noqa: E402
from src.models.npr_resnet import build_model  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising accuracy on the given scores (ties not split)."""
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    n = s.size
    n_pos = int(y.sum())
    n_neg = n - n_pos
    tp = np.cumsum(y == 1)
    pred_pos = np.arange(1, n + 1)
    correct = tp + (n_neg - (pred_pos - tp))
    valid = np.ones(n, dtype=bool)
    valid[:-1] = s[1:] != s[:-1]
    idx = np.flatnonzero(valid)
    best_k = int(np.argmax(correct[valid]))
    k = int(idx[best_k]) + 1
    if k >= n:
        return float(s[-1])
    # cut between s[k-1] and s[k]: any value in between works, prefer the midpoint
    return float(0.5 * (s[k - 1] + s[k])) if k > 0 else float(s[0] + 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "eval_256_nocrop.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(REPO_ROOT, "..", "database", "weights", "NPR.pth"))
    ap.add_argument("--test-root", default=None)
    ap.add_argument("--limit-per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=70)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "calibration.csv"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    # torch workers need named pipes, which the sandbox forbids; single-process
    # loading is fine for a few thousand images
    cfg.setdefault("data", {})["workers"] = args.workers
    test_root = args.test_root or cfg["data"]["test_root"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(cfg).to(device)
    info = load_checkpoint(model, args.ckpt, device=device)
    model.eval()
    print(f"[ckpt] {info['n_loaded']}/{info['n_in_checkpoint']} tensors from {os.path.basename(args.ckpt)}")

    generators = sorted(
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d, "real"))
    )
    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, object]] = []
    pooled_tune_s: List[np.ndarray] = []
    pooled_tune_y: List[np.ndarray] = []
    pooled_eval_s: List[np.ndarray] = []
    pooled_eval_y: List[np.ndarray] = []
    for gen in generators:
        loader, _ds = build_eval_loader(
            os.path.join(test_root, gen), cfg, batch_size=None,
            limit_per_class=args.limit_per_class,
        )
        labels, scores = collect_scores(model, loader, device, max_batches=None, progress=False)
        labels = labels.astype(np.int64)
        # split each class in half so tuning and evaluation never share an image
        tune_idx, eval_idx = [], []
        for cls in (0, 1):
            idx = np.flatnonzero(labels == cls)
            rng.shuffle(idx)
            half = idx.size // 2
            tune_idx.extend(idx[:half].tolist())
            eval_idx.extend(idx[half:].tolist())
        tune_idx = np.array(sorted(tune_idx))
        eval_idx = np.array(sorted(eval_idx))

        thr = best_threshold(scores[tune_idx], labels[tune_idx])
        pooled_tune_s.append(scores[tune_idx])
        pooled_tune_y.append(labels[tune_idx])
        pooled_eval_s.append(scores[eval_idx])
        pooled_eval_y.append(labels[eval_idx])
        row = {
            "generator": gen,
            "family": _family(gen),
            "n": int(labels.size),
            "acc_thr0": round(accuracy(labels[eval_idx], scores[eval_idx], 0.0), 4),
            "acc_tuned": round(accuracy(labels[eval_idx], scores[eval_idx], thr), 4),
            "threshold": round(thr, 4),
            "auc": round(auc(labels[eval_idx], scores[eval_idx]), 4),
        }
        rows.append(row)
        print(f"{gen:<26}{row['family']:<10} acc@0={row['acc_thr0']:.3f} "
              f"acc@tuned={row['acc_tuned']:.3f} (thr={thr:+.3f}) auc={row['auc']:.4f}")

    def mean(key: str, family: str | None = None) -> float:
        vals = [r[key] for r in rows if family is None or r["family"] == family]
        return float(np.mean(vals)) if vals else float("nan")

    print()
    print(f"{'group':<14}{'acc@0':>8}{'acc@tuned':>11}{'gain':>8}{'auc':>8}")
    for label, fam in (("all", None), ("gan", "gan"), ("diffusion", "diffusion")):
        a0, at, au = mean("acc_thr0", fam), mean("acc_tuned", fam), mean("auc", fam)
        print(f"{label:<14}{a0:>8.2f}{at:>11.2f}{at - a0:>+8.2f}{au:>8.2f}")

    # One threshold for everything is what you can actually deploy when the
    # source is unknown; per-generator tuning is an upper bound, not a result.
    if pooled_eval_s:
        ts = np.concatenate(pooled_tune_s)
        ty = np.concatenate(pooled_tune_y)
        es = np.concatenate(pooled_eval_s)
        ey = np.concatenate(pooled_eval_y)
        g_thr = best_threshold(ts, ty)
        print(
            f"\nsingle global threshold (tuned on all tuning halves): "
            f"thr={g_thr:+.3f}  acc@0={100 * accuracy(ey, es, 0.0):.2f}  "
            f"acc@thr={100 * accuracy(ey, es, g_thr):.2f}  auc={100 * auc(ey, es):.2f}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # machine-readable summary: the report reads this instead of hard-coding numbers
    summary = {
        "limit_per_class": args.limit_per_class,
        "seed": args.seed,
        "n_generators": len(rows),
        "groups": {
            label: {
                "acc_thr0": 100.0 * mean("acc_thr0", fam),
                "acc_tuned": 100.0 * mean("acc_tuned", fam),
                "auc": 100.0 * mean("auc", fam),
                "gain": 100.0 * (mean("acc_tuned", fam) - mean("acc_thr0", fam)),
            }
            for label, fam in (("all", None), ("gan", "gan"), ("diffusion", "diffusion"))
        },
    }
    if pooled_eval_s:
        ts = np.concatenate(pooled_tune_s)
        ty = np.concatenate(pooled_tune_y)
        es = np.concatenate(pooled_eval_s)
        ey = np.concatenate(pooled_eval_y)
        g_thr = best_threshold(ts, ty)
        summary["global"] = {
            "threshold": float(g_thr),
            "acc_thr0": 100.0 * accuracy(ey, es, 0.0),
            "acc_tuned": 100.0 * accuracy(ey, es, g_thr),
            "auc": 100.0 * auc(ey, es),
            "gain": 100.0 * (accuracy(ey, es, g_thr) - accuracy(ey, es, 0.0)),
        }
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"wrote {args.out} ({len(rows)} generators) and its _summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
