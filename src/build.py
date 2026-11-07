"""Config loading and DataLoader construction."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import yaml
from torch.utils.data import ConcatDataset, DataLoader

from .data.datasets import BinaryForensicsDataset
from .data.transforms import build_eval_transform, build_train_transform

__all__ = [
    "REPO_ROOT",
    "load_config",
    "merged_configs",
    "build_train_loader",
    "build_val_loader",
    "build_eval_loader",
    "build_multi_loader",
]

# src/build.py -> src -> <repo root>. Two levels, not three: three would point
# at the parent of the project and every relative dataset path would miss.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")


def load_config(path: Optional[str] = None) -> dict:
    """Load a YAML config; ``None`` means ``configs/default.yaml``."""
    if path is None:
        path = os.path.join(CONFIG_DIR, "default.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["__config_path__"] = os.path.abspath(path)
    for key in ("train_root", "val_root", "test_root"):
        v = cfg.get("data", {}).get(key)
        if v and not os.path.isabs(v):
            cfg["data"][key] = os.path.normpath(os.path.join(REPO_ROOT, v))
    return cfg


def merged_configs(base: dict, overrides: Optional[dict] = None) -> dict:
    """Shallow-per-section merge used by the sweep driver."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def _loader_kwargs(cfg: dict, train: bool, batch_size: Optional[int] = None) -> dict:
    d = cfg.get("data", {})
    # `workers` is the key used by the configs and the CLI flags; `num_workers`
    # is accepted as an alias so hand-written configs do not silently fall
    # back to the default of 4
    nw = int(d.get("workers", d.get("num_workers", 4)))
    return dict(
        batch_size=int(batch_size if batch_size is not None
                       else d.get("batch_size", 32)),
        shuffle=train,
        num_workers=nw,
        pin_memory=bool(d.get("pin_memory", True)),
        drop_last=train,
        persistent_workers=nw > 0,
    )


def _subdirs(root: str) -> list:
    return [os.path.join(root, g) for g in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, g))]


def build_train_loader(cfg: dict) -> DataLoader:
    """Training split: ``<root>/<generator>/{real,fake}``.

    Every generator folder uses the pinned real->0 / fake->1 mapping and the
    folders are concatenated, so the model sees one binary task rather than a
    generator-classification task.
    """
    root = cfg["data"]["train_root"]
    transform = build_train_transform(cfg)
    parts = [BinaryForensicsDataset(p, transform) for p in _subdirs(root)]
    if not parts:
        raise FileNotFoundError(f"no generator folders under {root}")
    return DataLoader(ConcatDataset(parts), **_loader_kwargs(cfg, train=True))


def build_val_loader(cfg: dict) -> DataLoader:
    root = cfg["data"]["val_root"]
    transform = build_eval_transform(cfg)
    parts = [BinaryForensicsDataset(p, transform) for p in _subdirs(root)]
    if not parts:
        raise FileNotFoundError(f"no generator folders under {root}")
    return DataLoader(ConcatDataset(parts), **_loader_kwargs(cfg, train=False))


def build_eval_loader(
    root: str,
    cfg: dict,
    degradation: Optional[dict] = None,
    batch_size: Optional[int] = None,
) -> Tuple[DataLoader, BinaryForensicsDataset]:
    """Evaluation loader for one ``<generator>/{real,fake}`` folder pair."""
    data_cfg = cfg.get("data", {})
    limit = data_cfg.get("eval_limit_per_class")
    dset = BinaryForensicsDataset(root, build_eval_transform(cfg, degradation),
                                 limit_per_class=int(limit) if limit else None)
    return DataLoader(dset, **_loader_kwargs(cfg, train=False,
                                             batch_size=batch_size)), dset


def build_multi_loader(
    root: str,
    cfg: dict,
    degradation: Optional[dict] = None,
    batch_size: Optional[int] = None,
) -> Tuple[DataLoader, ConcatDataset]:
    """One loader over every generator folder under ``root``."""
    transform = build_eval_transform(cfg, degradation)
    parts = [BinaryForensicsDataset(p, transform) for p in _subdirs(root)]
    if not parts:
        raise FileNotFoundError(f"no generator folders under {root}")
    dset = ConcatDataset(parts)
    return DataLoader(dset, **_loader_kwargs(cfg, train=False,
                                             batch_size=batch_size)), dset
