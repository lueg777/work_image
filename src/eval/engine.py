"""Shared evaluation engine: checkpoint loading and score collection."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..utils import get_device, logits_to_score
from .metrics import binary_metrics

__all__ = [
    "load_checkpoint",
    "collect_scores",
    "evaluate_loader",
    "summarise",
]


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd):
        return {k[len("module."):] if k.startswith("module.") else k: v
                for k, v in sd.items()}
    return dict(sd)


def load_checkpoint(model: nn.Module, path: str,
                    device: Optional[torch.device] = None,
                    strict: bool = False) -> Dict:
    """Load an NPR-style checkpoint, tolerating the usual wrapper formats.

    Accepts a bare ``state_dict`` or one nested under ``model`` / ``state_dict``
    / ``net``, and strips a ``DataParallel`` ``module.`` prefix.

    With ``strict=False`` a partial load still returns a report instead of
    raising: that is what makes it safe to load the official ``NPR.pth`` into a
    model whose ``fc1`` was rebuilt with a different ``num_classes``.
    """
    if device is None:
        device = next(model.parameters()).device
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "net", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a state_dict, got {type(obj)}")
    sd = _strip_module_prefix(obj)

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    report = {
        "path": os.path.abspath(path),
        "n_loaded": len(sd) - len(unexpected),
        "n_in_checkpoint": len(sd),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
    if strict and (missing or unexpected):
        raise RuntimeError(f"strict load failed: {report}")
    return report


@torch.no_grad()
def collect_scores(
    model: nn.Module,
    loader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model and return ``(labels, fake_probability)`` as numpy arrays."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    labels: List[np.ndarray] = []
    scores: List[np.ndarray] = []

    it = loader
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(loader, desc="eval", leave=False)
        except ImportError:
            pass

    for i, (imgs, targets) in enumerate(it):
        if max_batches is not None and i >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        scores.append(logits_to_score(logits).detach().cpu().numpy())
        labels.append(np.asarray(targets).reshape(-1))

    if not labels:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    return (np.concatenate(labels).astype(np.int64),
            np.concatenate(scores).astype(np.float64))


def evaluate_loader(
    model: nn.Module,
    loader,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    max_batches: Optional[int] = None,
    progress: bool = False,
) -> Dict[str, float]:
    y, s = collect_scores(model, loader, device=device,
                          max_batches=max_batches, progress=progress)
    m = binary_metrics(y, s, threshold=threshold)
    return m


def summarise(rows: Sequence[Dict], keys: Sequence[str] = ("acc", "auc", "ap")) -> Dict[str, float]:
    """Macro mean over per-generator rows, NaN-safe."""
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None and not np.isnan(r[k])]
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else float("nan")
    out["n_generators"] = float(len(rows))
    return out
