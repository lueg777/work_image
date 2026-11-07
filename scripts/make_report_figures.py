"""Draw the figures the report needs.

Three pictures, each answering one question that tables answer badly:

  fig_spectrum.png  what does the residual operator see?  (radial power spectrum,
                    real vs fake, split into GAN and diffusion groups)
  fig_ablation.png  how much does each residual operator cost?  (AUC per operator)
  fig_robust.png    what survives post-processing?  (AUC per attack)

Axis text is English on purpose: matplotlib's CJK fallback depends on the host's
installed fonts, and a report figure full of tofu boxes is worse than English
labels. The Chinese explanation lives in the caption.

Usage:
    python scripts/make_report_figures.py [--fig-dir figs] [--spectrum-per-side 160]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from src.eval.spectrum import radial_power_profile  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "datasets", "Generalization_Test")
SWEEP_JSON = os.path.join(REPO_ROOT, "runs", "sweep", "summary.json")
SWEEP_CSV = os.path.join(REPO_ROOT, "runs", "sweep", "cross_all_ops.csv")
ROBUST_CSV = os.path.join(REPO_ROOT, "runs", "real", "npr256", "robust_npr_nearest.csv")

# operator -> the group it belongs to; order is the one used in the report
OP_ORDER = [
    ("npr_nearest", "matched"),
    ("npr_bilinear", "interpolating"),
    ("npr_bicubic", "interpolating"),
    ("npr_box", "interpolating"),
    ("laplacian", "pure high-pass"),
    ("dct_lowpass", "pure high-pass"),
    ("fft_radial", "pure high-pass"),
    ("identity", "control"),
]
GROUP_COLOR = {
    "matched": "#1f4e79",
    "interpolating": "#c55a11",
    "pure high-pass": "#2e7d32",
    "control": "#7f7f7f",
}
DIFFUSION_TAGS = ("adm", "glide", "midjourney", "sdv", "sdxl", "vqdm", "vqdiffusion",
                  "wukong", "dalle", "ddpm", "iddpm", "diffusion", "coco")


def _family(name: str) -> str:
    low = "".join(ch for ch in name.lower() if ch.isalnum())
    return "diffusion" if any(t in low for t in DIFFUSION_TAGS) else "gan"


def _image_files(side_dir: str) -> List[str]:
    out = []
    for entry in sorted(os.listdir(side_dir)):
        path = os.path.join(side_dir, entry)
        if os.path.isfile(path) and os.path.splitext(entry)[1].lower() in {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
        }:
            out.append(path)
    return out


def spectrum_curves(per_side: int, seed: int = 70, size: int = 256) -> Tuple[Dict[str, np.ndarray], int]:
    """Average radial power profile for (family, label) pairs.

    Every image is resampled to ``size`` x ``size`` first. The staged test set has
    real images at ~500x375 (and 1024x1024 for one generator) against 256/512/1024
    square fakes, so comparing spectra at native resolution would mostly measure
    the resolution difference -- which is collinear with the label. Resampling to a
    common grid is the cheapest way to make the two sides comparable; the caption
    says so, because the resampling does not undo the original JPEG history.
    """
    rng = random.Random(seed)
    buckets: Dict[str, List[np.ndarray]] = defaultdict(list)

    for gen in sorted(os.listdir(DATA_ROOT)):
        gen_dir = os.path.join(DATA_ROOT, gen)
        if not os.path.isdir(os.path.join(gen_dir, "real")):
            continue
        fam = _family(gen)
        files_real = _image_files(os.path.join(gen_dir, "real"))
        files_fake = _image_files(os.path.join(gen_dir, "fake")) if os.path.isdir(
            os.path.join(gen_dir, "fake")) else []
        if not files_real or not files_fake:
            continue
        n = max(4, per_side // 10)
        for label, files in (("real", files_real), ("fake", files_fake)):
            pick = files if len(files) <= n else rng.sample(files, n)
            for path in pick:
                try:
                    with Image.open(path) as im:
                        im = im.convert("RGB").resize((size, size), Image.BICUBIC)
                        arr = np.asarray(im, dtype=np.float32) / 255.0
                except Exception:
                    continue
                prof = radial_power_profile(arr, n_bins=64, r_max=0.5)
                buckets[f"{fam}_{label}"].append(prof)

    curves = {k: np.mean(np.stack(v), axis=0) for k, v in buckets.items() if v}
    return curves, sum(len(v) for v in buckets.values())


def fig_spectrum(fig_dir: str, per_side: int) -> str:
    curves, n = spectrum_curves(per_side)
    if not curves:
        raise SystemExit(f"no images under {DATA_ROOT} - run prepare_data.py first")
    bins = len(next(iter(curves.values())))
    xs = np.linspace(0.0, 0.5, bins)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
    style = {
        "gan_real": ("#7f7f7f", "-"),
        "gan_fake": ("#7f7f7f", "--"),
        "diffusion_real": ("#1f4e79", "-"),
        "diffusion_fake": ("#1f4e79", "--"),
    }
    labels = {
        "gan_real": "GAN real",
        "gan_fake": "GAN fake",
        "diffusion_real": "Diffusion real",
        "diffusion_fake": "Diffusion fake",
    }
    for key in ("gan_real", "gan_fake", "diffusion_real", "diffusion_fake"):
        if key not in curves:
            continue
        color, ls = style[key]
        ax.semilogy(xs[1:], curves[key][1:], ls, color=color, lw=1.5, label=labels[key])
    ax.set_xlabel("normalised radius  (0.5 = Nyquist)", fontsize=8.5)
    ax.set_ylabel("log mean power (azimuthal average)", fontsize=8.5)
    ax.set_title(f"Radial power spectrum, {n} images resampled to 256$^2$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    # the interesting part is the sign and size of fake-minus-real, not the
    # absolute level, so plot the difference on its own panel
    for fam, color in (("gan", "#7f7f7f"), ("diffusion", "#1f4e79")):
        rk, fk = f"{fam}_real", f"{fam}_fake"
        if rk in curves and fk in curves:
            ax2.plot(xs[6:], (curves[fk] - curves[rk])[6:], "-", color=color, lw=1.5, label=fam)
    ax2.axhline(0.0, color="#b00020", lw=0.9, ls=":")
    ax2.set_xlabel("normalised radius", fontsize=8.5)
    ax2.set_ylabel("mean power  fake − real", fontsize=8.5)
    ax2.set_title("Where the fake side departs from real", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(fig_dir, "fig_spectrum.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def _load_sweep() -> Dict[str, Dict[str, float]]:
    if os.path.isfile(SWEEP_JSON):
        with open(SWEEP_JSON, encoding="utf-8") as fh:
            raw = json.load(fh)
        summary = raw.get("summary", raw)
        out = {}
        for op, row in summary.items():
            if isinstance(row, dict):
                out[op] = {
                    "auc": float(row.get("mean_auc", row.get("auc", 0.0))) * 100.0,
                    "acc": float(row.get("mean_acc", row.get("acc", 0.0))) * 100.0,
                }
        if out:
            return out
    if os.path.isfile(SWEEP_CSV):
        acc: Dict[str, List[float]] = defaultdict(list)
        with open(SWEEP_CSV, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = row.get("op") or row.get("operator") or row.get("name")
                acc[name].append(float(row["acc"]))
        return {k: {"acc": 100.0 * float(np.mean(v)), "auc": float("nan")} for k, v in acc.items()}
    raise SystemExit("no sweep results: run scripts/sweep first")


def fig_ablation(fig_dir: str) -> str:
    sweep = _load_sweep()
    names, aucs, colors = [], [], []
    for op, group in OP_ORDER:
        row = sweep.get(op)
        if row is None:
            continue
        names.append(op)
        aucs.append(row["auc"])
        colors.append(GROUP_COLOR[group])
    base = aucs[names.index("npr_nearest")] if "npr_nearest" in names else max(aucs)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.bar(range(len(names)), aucs, color=colors, width=0.68)
    for i, (bar, val) in enumerate(zip(bars, aucs)):
        delta = val - base
        txt = f"{val:.1f}" if i == names.index("npr_nearest") else f"{val:.1f}\n({delta:+.1f})"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, txt,
                ha="center", va="bottom", fontsize=7.6)
    ax.axhline(50, color="#b00020", lw=0.9, ls=":")
    ax.text(0.06, 52.0, "chance", color="#b00020", fontsize=7.5, ha="left")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("AUC (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
    ax.legend(handles, list(GROUP_COLOR.keys()), frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Operator ablation, same weights (17 generators, 300/class)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(fig_dir, "fig_ablation.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def fig_robust(fig_dir: str) -> str:
    if not os.path.isfile(ROBUST_CSV):
        raise SystemExit(f"missing {ROBUST_CSV}")
    buckets: Dict[str, List[float]] = defaultdict(list)
    with open(ROBUST_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            buckets[row["attack"]].append(float(row["auc"]) * 100.0)
    order = ["none", "jpeg95", "jpeg75", "jpeg50", "resize05"]
    order = [k for k in order if k in buckets]
    vals = [float(np.mean(buckets[k])) for k in order]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    colors = ["#1f4e79"] + ["#c55a11"] * (len(order) - 1)
    bars = ax.bar(range(len(order)), vals, color=colors, width=0.62)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(50, color="#b00020", lw=0.9, ls=":")
    ax.text(0.06, 52.0, "chance", color="#b00020", fontsize=7.5, ha="left")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("AUC (%)")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Robustness, same weights (17 generators, 100/class)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(fig_dir, "fig_robust.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig-dir", default=os.path.join(REPO_ROOT, "figs"))
    ap.add_argument("--spectrum-per-side", type=int, default=160)
    ap.add_argument("--skip-spectrum", action="store_true")
    ap.add_argument("--seed", type=int, default=70)
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    made = []
    if not args.skip_spectrum:
        made.append(fig_spectrum(args.fig_dir, args.spectrum_per_side))
    made.append(fig_ablation(args.fig_dir))
    made.append(fig_robust(args.fig_dir))
    for path in made:
        print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
