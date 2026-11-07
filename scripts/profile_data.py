"""Profile the staged test set: what the real and fake sides actually look like.

The point is not curiosity. AIGI_test mixes several sources, and if the container
format or the resolution is perfectly correlated with the label, then any
real-vs-fake difference the model picks up could be a source artefact rather than
a generator fingerprint. This script measures that correlation so the report can
state it instead of hand-waving about it.

Usage:
    python scripts/profile_data.py --root datasets/Generalization_Test \
        --out results/data_profile.csv [--samples 120]
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layout import iter_image_files  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _median(values: List[int]) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return 0.5 * (values[mid - 1] + values[mid])


def profile_generator(gen_dir: str, samples: int, rng: random.Random) -> Dict[str, object]:
    out: Dict[str, object] = {"generator": os.path.basename(gen_dir)}
    for label in ("real", "fake"):
        side = os.path.join(gen_dir, label)
        if not os.path.isdir(side):
            continue
        files = iter_image_files(side)
        exts = Counter(os.path.splitext(f)[1].lower() for f in files)
        total = sum(exts.values()) or 1
        top_ext, top_n = (exts.most_common(1)[0] if exts else ("", 0))
        out[f"{label}_n"] = len(files)
        out[f"{label}_ext"] = top_ext
        out[f"{label}_ext_share"] = round(100.0 * top_n / total, 1)
        out[f"{label}_next_ext"] = (exts.most_common(2)[1][0] if len(exts) > 1 else "")

        pick = files if len(files) <= samples else rng.sample(files, samples)
        widths, heights = [], []
        for path in pick:
            try:
                with Image.open(path) as im:
                    widths.append(im.width)
                    heights.append(im.height)
            except Exception:
                continue
        out[f"{label}_w"] = _median(widths)
        out[f"{label}_h"] = _median(heights)
        out[f"{label}_square"] = (
            round(100.0 * sum(1 for w, h in zip(widths, heights) if w == h) / max(len(widths), 1), 1)
        )
    return out


FIELDS = [
    "generator", "family",
    "real_n", "real_ext", "real_ext_share", "real_next_ext", "real_w", "real_h", "real_square",
    "fake_n", "fake_ext", "fake_ext_share", "fake_next_ext", "fake_w", "fake_h", "fake_square",
]


def _family(name: str) -> str:
    from src.eval.cross_generator import _family as fam  # single source of truth

    return fam(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "datasets", "Generalization_Test"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "data_profile.csv"))
    ap.add_argument("--samples", type=int, default=120, help="images sampled per side for size stats")
    ap.add_argument("--seed", type=int, default=70)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"[error] not a directory: {args.root}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rows: List[Dict[str, object]] = []
    for name in sorted(os.listdir(args.root)):
        gen_dir = os.path.join(args.root, name)
        if not os.path.isdir(gen_dir):
            continue
        if not os.path.isdir(os.path.join(gen_dir, "real")):
            continue
        row = profile_generator(gen_dir, args.samples, rng)
        row["family"] = _family(name)
        rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})

    hdr = f"{'generator':<26}{'fam':<10}{'real ext':<10}{'%':>6}{'real size':>14}{'fake ext':<10}{'%':>6}{'fake size':>14}"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        rs = f"{row.get('real_w')}x{row.get('real_h')}"
        fs = f"{row.get('fake_w')}x{row.get('fake_h')}"
        print(
            f"{str(row['generator']):<26}{str(row['family']):<10}"
            f"{str(row.get('real_ext')):<10}{row.get('real_ext_share'):>6}{rs:>14}"
            f"{str(row.get('fake_ext')):<10}{row.get('fake_ext_share'):>6}{fs:>14}"
        )
    print(f"\nwrote {args.out} ({len(rows)} generators)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
