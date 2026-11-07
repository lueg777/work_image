"""Cross-generator generalisation sweep.

The core experiment. A detector is trained on the ForenSynths families
(``car, cat, chair, horse``: ProGAN / StyleGAN / BigGAN / CycleGAN-ish
sources) and then evaluated on generators it has never seen. The interesting
quantity is not the mean accuracy but the **shape of the decay per generator**,
which is where the operator-matching hypothesis makes falsifiable predictions:

*   if NPR generalises because of its high-pass operator, then an
    operator that matches the decoder's own up-sampling should transfer
    better on diffusion models than on GANs (or vice versa, depending on
    which up-sampling kernel dominates);
*   if it generalises simply because it is high-pass, every high-pass
    variant should land within noise of each other.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..build import build_eval_loader, REPO_ROOT
from ..data.datasets import list_generators
from ..utils import write_csv
from .engine import evaluate_loader, summarise

__all__ = ["run_cross_generator", "format_table"]

# Diffusion-family generators get flagged so the report can split GAN vs diffusion.
DIFFUSION_TAGS = {
    "ADM", "Glide", "Midjourney", "SDv1.4", "SDv1.5", "SDv2.1", "VQDM",
    "Wukong", "DALLE2", "ddpm", "iddpm", "sdv4", "sdv5", "sdxl", "vqdiffusion",
}


def _family(generator: str) -> str:
    return "diffusion" if generator in DIFFUSION_TAGS else "gan"


def run_cross_generator(
    model: nn.Module,
    test_root: str,
    cfg: dict,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    max_batches: Optional[int] = None,
    op_name: str = "",
    out_csv: Optional[str] = None,
    progress: bool = True,
) -> Dict:
    """Evaluate ``model`` on every ``<generator>/{real,fake}`` under ``test_root``."""
    if device is None:
        device = next(model.parameters()).device
    if not os.path.isdir(test_root):
        raise FileNotFoundError(test_root)

    rows: List[Dict] = []
    for gen in list_generators(test_root):
        root = os.path.join(test_root, gen)
        try:
            loader, dset = build_eval_loader(root, cfg)
        except RuntimeError as e:
            print(f"[skip] {gen}: {e}")
            continue
        m = evaluate_loader(model, loader, device=device, threshold=threshold,
                            max_batches=max_batches, progress=progress)
        row = {"generator": gen, "family": _family(gen), "op": op_name,
               "n_real": sum(1 for _, t in dset.samples if t == 0),
               "n_fake": sum(1 for _, t in dset.samples if t == 1)}
        row.update(m)
        rows.append(row)
        print(f"  {gen:<12} acc={m['acc']*100:5.1f}  auc={m['auc']*100:5.1f}  "
              f"ap={m['ap']*100:5.1f}  n={int(m['n'])}")

    summary = summarise(rows)
    for fam in ("gan", "diffusion"):
        sub = [r for r in rows if r["family"] == fam]
        if sub:
            s = summarise(sub)
            for k, v in s.items():
                summary[f"{fam}_{k}"] = v

    result = {"rows": rows, "summary": summary, "op": op_name,
              "test_root": os.path.abspath(test_root)}
    if out_csv:
        write_csv(rows, out_csv)
        print(f"[write] {out_csv}")
    return result


def format_table(rows: List[Dict], metric: str = "acc") -> str:
    """Plain-text table, sorted by generator name, for pasting into the report."""
    if not rows:
        return "(no rows)"
    w = max(len(r["generator"]) for r in rows) + 2
    head = f"{'generator':<{w}}{'family':<12}{metric:>8}"
    lines = [head, "-" * len(head)]
    for r in sorted(rows, key=lambda x: x["generator"]):
        v = r.get(metric)
        cell = f"{v*100:7.2f}" if isinstance(v, float) and v == v else "    nan"
        lines.append(f"{r['generator']:<{w}}{r['family']:<12}{cell}")
    return "\n".join(lines)
