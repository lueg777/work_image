"""Small shared utilities: seeding, devices, IO, score conversion."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "seed_everything",
    "get_device",
    "ensure_dir",
    "timestamp",
    "save_json",
    "load_json",
    "write_csv",
    "append_csv",
    "AverageMeter",
    "logits_to_score",
    "score_to_label",
]


def seed_everything(seed: int = 70) -> None:
    """Seed python/numpy/torch. 70 is the seed the NPR README uses for GenImage."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: Sequence[Dict], path: str,
              fieldnames: Optional[Sequence[str]] = None,
              append: bool = False) -> None:
    rows = list(rows)
    if not rows:
        return
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    mode = "a" if append and os.path.exists(path) else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        if mode == "w":
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def append_csv(row: Dict, path: str) -> None:
    write_csv([row], path, append=True)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else float("nan")


def logits_to_score(logits: torch.Tensor) -> torch.Tensor:
    """Turn raw model output into P(fake) in [0, 1].

    * 1 output -> sigmoid (the NPR head, trained with BCEWithLogits)
    * 2 outputs -> softmax, taking the index-1 (fake) column
    """
    if logits.dim() > 1 and logits.size(1) > 1:
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits.reshape(-1))


def score_to_label(scores: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (scores >= threshold).long()
