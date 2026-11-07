"""Robustness sweep: does the operator advantage survive post-processing?

Two families of attack, both applied to the PIL image before ``ToTensor`` so
the tensor the network sees is exactly what a re-encoded file would contain:

*   JPEG re-compression at quality 95 / 75 / 50 -- the standard "was it posted
    to a social network" test. JPEG quantises DCT coefficients, i.e. it attacks
    precisely the frequency band a spectral fingerprint would live in. If a
    detector's edge comes from high-frequency artefacts, JPEG 50 should erase it.
*   0.5x down/up rescale -- destroys the fine sampling grid that up-sampling
    fingerprints live on, while largely preserving global content.

Run order matters for cost: ``none`` first so a broken setup fails fast.
"""

from __future__ import annotations

import os
from itertools import product
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..build import build_eval_loader
from ..data.datasets import list_generators
from ..utils import write_csv
from .engine import evaluate_loader, summarise
from .cross_generator import _family

__all__ = ["ATTACKS", "run_robustness", "format_matrix"]

# (name, degradation-spec or None)
ATTACKS: List[tuple] = [
    ("none", None),
    ("jpeg95", {"kind": "jpeg", "quality": 95}),
    ("jpeg75", {"kind": "jpeg", "quality": 75}),
    ("jpeg50", {"kind": "jpeg", "quality": 50}),
    ("resize05", {"kind": "resize", "factor": 0.5}),
]


def run_robustness(
    model: nn.Module,
    test_root: str,
    cfg: dict,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    max_batches: Optional[int] = None,
    op_name: str = "",
    attacks: Optional[List[tuple]] = None,
    generators: Optional[List[str]] = None,
    out_csv: Optional[str] = None,
    progress: bool = True,
) -> Dict:
    if device is None:
        device = next(model.parameters()).device
    attacks = list(attacks or ATTACKS)
    gens = list(generators or list_generators(test_root))

    rows: List[Dict] = []
    for gen, (name, spec) in product(gens, attacks):
        root = os.path.join(test_root, gen)
        if not os.path.isdir(root):
            print(f"[skip] missing {root}")
            continue
        try:
            loader, _ = build_eval_loader(root, cfg, degradation=spec)
        except RuntimeError as e:
            print(f"[skip] {gen}/{name}: {e}")
            continue
        m = evaluate_loader(model, loader, device=device, threshold=threshold,
                            max_batches=max_batches, progress=progress)
        row = {"generator": gen, "family": _family(gen), "attack": name,
               "op": op_name}
        row.update(m)
        rows.append(row)
        print(f"  {gen:<12} {name:<9} acc={m['acc']*100:5.1f} "
              f"auc={m['auc']*100:5.1f} ap={m['ap']*100:5.1f}")

    # per-attack macro means, which is what the report table actually shows
    summary = {}
    for name, _ in attacks:
        sub = [r for r in rows if r["attack"] == name]
        if sub:
            summary[name] = summarise(sub)
    result = {"rows": rows, "summary": summary, "op": op_name,
              "test_root": os.path.abspath(test_root)}
    if out_csv:
        write_csv(rows, out_csv)
        print(f"[write] {out_csv}")
    return result


def format_matrix(rows: List[Dict], metric: str = "acc") -> str:
    """Full generator x attack matrix, one block per metric."""
    if not rows:
        return "(no rows)"
    gens = sorted({r["generator"] for r in rows})
    atks = []
    for r in rows:
        if r["attack"] not in atks:
            atks.append(r["attack"])
    idx = {(r["generator"], r["attack"]): r.get(metric) for r in rows}

    w = max(len(g) for g in gens) + 2
    head = f"{'generator':<{w}}" + "".join(f"{a:>10}" for a in atks)
    lines = [f"[{metric}]", head, "-" * len(head)]
    for g in gens:
        cells = []
        for a in atks:
            v = idx.get((g, a))
            cells.append(f"{v*100:10.2f}" if isinstance(v, float) and v == v
                         else f"{'nan':>10}")
        lines.append(f"{g:<{w}}" + "".join(cells))
    means = []
    for a in atks:
        vals = [idx[(g, a)] for g in gens
                if isinstance(idx.get((g, a)), float) and idx[(g, a)] == idx[(g, a)]]
        means.append(f"{np.mean(vals)*100:10.2f}" if vals else f"{'nan':>10}")
    lines.append(f"{'MEAN':<{w}}" + "".join(means))
    return "\n".join(lines)
