"""Stage, check and inventory the AIGC-forensics datasets.

The published benchmark zips are laid out for *their* training script, in three
different conventions (``0_real``/``1_fake``, ``real``/``fake``, nested
``<benchmark>/<generator>/...``). Our loader wants exactly one shape::

    <root>/<generator>/real/<image>
    <root>/<generator>/fake/<image>

This script converts one into the other, then double-checks the result. It is
torch-free and PIL-free on purpose, so it runs before the training environment
exists (bundled python is enough).

Commands
--------
``organize``  copy/hard-link a downloaded tree into the canonical layout
``verify``    counts, imbalance, duplicates, and train/test leakage
``manifest``  CSV inventory (path, generator, label, bytes, sha1) for the report

Examples
--------
::

    # ForenSynths train: <class>/{0_real,1_fake} -- every image is ProGAN
    python scripts/prepare_data.py organize \\
        --src D:/downloads/ForenSynths_train_val/train \\
        --dst datasets/ForenSynths_train_val/train \\
        --generator ProGAN

    # Generalization_Test: keep benchmarks apart from each other
    python scripts/prepare_data.py organize \\
        --src D:/downloads/Generalization_Test \\
        --dst datasets/Generalization_Test --layout flat

    python scripts/prepare_data.py verify --root datasets/Generalization_Test \\
        --leakage-with datasets/ForenSynths_train_val/train
    python scripts/prepare_data.py manifest --root datasets/Generalization_Test \\
        --out manifests/generalization_test.csv

Everything defaults to read-only reporting except ``organize``, which never
overwrites: a name collision with different content is renamed ``__dup2`` and
counted, so a half-finished copy can be re-run safely.

``--mode auto`` (default) hard-links when the source and destination sit on the
same volume, which costs no extra disk for a 30GB dataset -- but a staged file
*is* the downloaded file, so editing or rewriting it in place edits the
original download too. Use ``--mode copy`` if the two trees must be independent.

Known trap this script exists to prevent: ``ForenSynths`` training folders are
named ``car``/``cat``/``chair``/``horse``. Those are *classes* of one generator
(ProGAN), not four generators. Staged as-is, the model would learn "which
object" instead of "synthetic or not", and the numbers would look plausible.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.layout import (  # noqa: E402
    file_sha1,
    is_image_file,
    iter_image_files,
    pair_class_dirs,
)

FORENSYNTHS_CLASSES = {"car", "cat", "chair", "horse"}


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def _subdirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def collect_pairs(src: str) -> List[Dict]:
    """Every real/fake directory pair under ``src``, generator dir included.

    A directory qualifies only when :func:`pair_class_dirs` succeeds on its
    children -- the same rule the loader applies, so this can never stage a
    layout the loader would reject.
    """
    pairs: List[Dict] = []
    for dirpath, dirnames, _ in os.walk(src):
        dirnames.sort()
        found = pair_class_dirs(dirnames)
        if found is None:
            continue
        real_name, fake_name, exact = found
        real_dir = os.path.join(dirpath, real_name)
        fake_dir = os.path.join(dirpath, fake_name)
        # A class folder holds images directly.  Two folders that merely *look*
        # like real/fake but contain sub-directories are benchmark containers
        # (`AIGCDetectBenchmark` / `UniversalFakeDetect` at the top of the
        # official zips), not a class pair -- guessing here would swallow the
        # whole tree into one bogus generator.
        if (not iter_image_files(real_dir) and not iter_image_files(fake_dir)
                and (os.path.isdir(real_dir) and os.path.isdir(fake_dir))
                and (_subdirs(real_dir) or _subdirs(fake_dir))):
            continue
        pairs.append({
            "gen_dir": dirpath,
            "real": real_dir,
            "fake": fake_dir,
            "exact": exact,
        })
        # never descend into the two class folders themselves
        dirnames[:] = [d for d in dirnames if d not in (real_name, fake_name)]
    return pairs


def iter_generators(root: str) -> List[Dict]:
    """Generators under ``root``, tolerating a flat or a nested layout.

    Flat: ``<root>/<generator>/{real,fake}``.
    Nested: ``<root>/<benchmark>/<generator>/{real,fake}`` (what the official
    zips ship). The nested form is reported as ``benchmark/generator`` so the
    two can never be conflated in a results table.
    """
    out: List[Dict] = []
    flat = 0
    for name in _subdirs(root):
        gpath = os.path.join(root, name)
        children = _subdirs(gpath)
        found = pair_class_dirs(children)
        if found is None:
            continue
        flat += 1
        out.append(_describe(name, gpath, children))
    if flat:
        return out
    for bench in _subdirs(root):
        bpath = os.path.join(root, bench)
        for name in _subdirs(bpath):
            gpath = os.path.join(bpath, name)
            children = _subdirs(gpath)
            if pair_class_dirs(children) is None:
                continue
            out.append(_describe(f"{bench}/{name}", gpath, children))
    return out


def _describe(display: str, gpath: str, children: List[str]) -> Dict:
    real_name, fake_name, exact = pair_class_dirs(children)  # type: ignore[misc]
    real_files = iter_image_files(os.path.join(gpath, real_name))
    fake_files = iter_image_files(os.path.join(gpath, fake_name))
    others = [c for c in children if c not in (real_name, fake_name)]
    stray = [f for f in sorted(os.listdir(gpath))
             if os.path.isfile(os.path.join(gpath, f)) and not is_image_file(f)]
    return {
        "name": display,
        "path": gpath,
        "exact": exact,
        "real_name": real_name,
        "fake_name": fake_name,
        "real_files": real_files,
        "fake_files": fake_files,
        "other_dirs": others,
        "stray_files": stray,
    }


# --------------------------------------------------------------------------- #
# organize
# --------------------------------------------------------------------------- #
def _benchmark_of(gen_dir: str, src: str) -> str:
    rel = os.path.relpath(os.path.normpath(gen_dir), os.path.normpath(src))
    rel = rel.replace(os.sep, "/")
    if rel == "." or "/" not in rel:
        return "_"
    return rel.rsplit("/", 1)[0].split("/")[-1]


def assign_targets(pairs: List[Dict], src: str, layout: str,
                   mapping: Dict[str, str], forced: Optional[str]) -> List[List[str]]:
    """Destination path components (under ``dst``) for every pair.

    ``flat`` puts every generator side by side, which is what
    ``src/eval/cross_generator.py`` expects. Two benchmarks that share a
    generator name (``ADM`` lives in two of them) would collide, so colliding
    names get a ``<benchmark>__`` prefix instead of a silent overwrite.
    """
    candidates: List[str] = []
    for p in pairs:
        rel = os.path.relpath(os.path.normpath(p["gen_dir"]), os.path.normpath(src))
        rel = rel.replace(os.sep, "/")
        base = os.path.basename(os.path.normpath(p["gen_dir"]))
        name = forced or mapping.get(rel) or mapping.get(base) or base
        candidates.append(name)

    if layout == "nested":
        out = []
        for p, name in zip(pairs, candidates):
            bench = _benchmark_of(p["gen_dir"], src)
            out.append([name] if bench == "_" else [bench, name])
        return out

    counts = Counter(candidates)
    out = []
    seen: Counter = Counter()
    for p, name in zip(pairs, candidates):
        if counts[name] > 1:
            bench = _benchmark_of(p["gen_dir"], src)
            if bench != "_":
                name = f"{bench}__{name}"
        seen[name] += 1
        if seen[name] > 1:
            name = f"{name}_{seen[name]}"
        out.append([name])
    return out


def _same_content(a: str, b: str) -> bool:
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    return file_sha1(a) == file_sha1(b)


def _unique_target(directory: str, filename: str, taken: set) -> str:
    stem, ext = os.path.splitext(filename)
    i = 2
    while True:
        cand = os.path.join(directory, f"{stem}__dup{i}{ext}")
        if not os.path.exists(cand) and cand not in taken:
            return cand
        i += 1


def _place(src: str, dst: str, mode: str) -> str:
    if mode in ("auto", "hardlink"):
        try:
            os.link(src, dst)
            return "linked"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(src, dst)
    return "copied"


def organize(src: str, dst: str, layout: str = "flat", mode: str = "auto",
             generator: Optional[str] = None, mapping: Optional[Dict[str, str]] = None,
             dry_run: bool = False, verbose: bool = False) -> Dict:
    if not os.path.isdir(src):
        raise SystemExit(f"--src is not a directory: {src}")
    pairs = collect_pairs(src)
    if not pairs:
        raise SystemExit(
            f"no <generator>/{{real,fake}} pairs under {src}\n"
            f"looked for subfolders named real/0_real/natural and fake/1_fake/synthetic"
        )

    targets = assign_targets(pairs, src, layout, mapping or {}, generator)
    stats: Counter = Counter()
    problems: List[str] = []

    for pair, comps in zip(pairs, targets):
        out_dir = os.path.join(dst, *comps)
        if not pair["exact"]:
            problems.append(f"{pair['gen_dir']}: class folders guessed "
                            f"({os.path.basename(pair['real'])}/"
                            f"{os.path.basename(pair['fake'])}), not name-matched")
        for side in ("real", "fake"):
            files = iter_image_files(pair[side])
            if not files:
                problems.append(f"{pair[side]}: no image files")
                continue
            target_dir = os.path.join(out_dir, side)
            for f in files:
                stats["files"] += 1
                target = os.path.join(target_dir, os.path.basename(f))
                if os.path.exists(target):
                    if _same_content(target, f):
                        stats["skipped"] += 1
                        continue
                    target = _unique_target(target_dir, os.path.basename(f), set())
                    stats["renamed"] += 1
                if dry_run:
                    stats["planned"] += 1
                    continue
                os.makedirs(target_dir, exist_ok=True)
                action = _place(f, target, mode)
                stats[action] += 1
                if verbose:
                    print(f"  {action}: {f} -> {target}")

    names = [c[0] if len(c) == 1 else "/".join(c) for c in targets]
    if layout == "flat" and not generator:
        classes = {n for n in names if n.lower() in FORENSYNTHS_CLASSES}
        if classes:
            problems.append(
                "generator names look like ForenSynths *classes*, not generators "
                f"({sorted(classes)}); every image in them is ProGAN. Re-run with "
                f"--generator ProGAN to stage them as one generator."
            )

    print(f"\n{'dry run: ' if dry_run else ''}organize {src} -> {dst}  [{layout}]")
    for comps in targets:
        print(f"  {'/'.join(comps)}")
    print(f"\n  pairs={len(pairs)} files={stats['files']} "
          f"linked={stats['linked']} copied={stats['copied']} "
          f"skipped(existing)={stats['skipped']} renamed={stats['renamed']} "
          f"planned={stats['planned']}")
    if problems:
        print("\n  WARNINGS")
        for p in problems:
            print(f"    - {p}")
    return {"stats": dict(stats), "problems": problems, "pairs": len(pairs),
            "generators": names}


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def _hash_all(paths: List[str], label: str, progress_every: int = 20000) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for i, p in enumerate(paths, start=1):
        mapping.setdefault(file_sha1(p), []).append(p)
        if progress_every and i % progress_every == 0:
            print(f"  .. hashed {i}/{len(paths)} {label} files")
    return mapping


def verify(root: str, leakage_with: Optional[str] = None, strict: bool = False,
           expect_real: Optional[int] = None, expect_fake: Optional[int] = None,
           max_ratio: float = 10.0, no_hash: bool = False) -> int:
    if not os.path.isdir(root):
        raise SystemExit(f"--root is not a directory: {root}")
    gens = iter_generators(root)
    if not gens:
        print(f"FAIL  no generators found under {root}")
        return 1

    errors: List[str] = []
    warnings: List[str] = []

    print(f"\n{'generator':34s} {'real':>8s} {'fake':>8s} {'ratio':>7s}  notes")
    print("-" * 78)
    tot_r = tot_f = 0
    for g in gens:
        nr, nf = len(g["real_files"]), len(g["fake_files"])
        tot_r += nr
        tot_f += nf
        notes = []
        if not g["exact"]:
            notes.append("guessed class names")
            warnings.append(f"{g['name']}: class names guessed, not name-matched")
        if g["other_dirs"]:
            notes.append(f"extra dirs {g['other_dirs']}")
            warnings.append(f"{g['name']}: extra subdirectories {g['other_dirs']}")
        if g["stray_files"]:
            notes.append(f"{len(g['stray_files'])} stray files")
            warnings.append(f"{g['name']}: non-image files {g['stray_files'][:3]}")
        if nr == 0:
            errors.append(f"{g['name']}: real side empty ({g['real_name']})")
        if nf == 0:
            errors.append(f"{g['name']}: fake side empty ({g['fake_name']})")
        if expect_real is not None and nr != expect_real:
            warnings.append(f"{g['name']}: real={nr}, expected {expect_real}")
        if expect_fake is not None and nf != expect_fake:
            warnings.append(f"{g['name']}: fake={nf}, expected {expect_fake}")
        ratio = (max(nr, nf) / max(min(nr, nf), 1)) if (nr or nf) else float("nan")
        if nr and nf and max(nr, nf) > max_ratio * min(nr, nf):
            warnings.append(f"{g['name']}: imbalanced {nr}/{nf} (>{max_ratio:.0f}x)")
        print(f"{g['name']:34s} {nr:8d} {nf:8d} {ratio:7.2f}  {', '.join(notes)}")
    print("-" * 78)
    print(f"{'TOTAL':34s} {tot_r:8d} {tot_f:8d} "
          f"{(max(tot_r, tot_f) / max(min(tot_r, tot_f), 1)):7.2f}  "
          f"{len(gens)} generators")

    if not no_hash:
        print("\n  hashing for duplicate detection...")
        seen: Dict[str, List[str]] = {}
        for g in gens:
            for p in g["real_files"] + g["fake_files"]:
                seen.setdefault(file_sha1(p), []).append(p)
        dups = {h: v for h, v in seen.items() if len(v) > 1}
        if dups:
            warnings.append(f"{len(dups)} duplicate image(s) inside {root} "
                            f"({sum(len(v) - 1 for v in dups.values())} redundant copies)")
            for h, v in list(dups.items())[:3]:
                print(f"    dup: {v[0]} == {v[1]}")
        else:
            print("    no duplicates inside the split")

        if leakage_with:
            print(f"  checking leakage against {leakage_with} ...")
            other_paths = []
            for g in iter_generators(leakage_with):
                other_paths += g["real_files"] + g["fake_files"]
            if not other_paths:
                warnings.append(f"{leakage_with}: no images found to compare against")
            else:
                other = _hash_all(other_paths, "reference")
                shared = set(seen) & set(other)
                if shared:
                    errors.append(
                        f"LEAKAGE: {len(shared)} image(s) appear in both "
                        f"{root} and {leakage_with}")
                    for h in list(shared)[:3]:
                        print(f"    leak: {seen[h][0]} == {other[h][0]}")
                else:
                    print(f"    no overlap across {len(other)} reference images")

    print()
    for e in errors:
        print(f"FAIL  {e}")
    for w in warnings:
        print(f"WARN  {w}")
    bad = bool(errors) or (strict and bool(warnings))
    print(f"RESULT  {'FAIL' if bad else 'OK'}  "
          f"({len(gens)} generators, {tot_r} real / {tot_f} fake, "
          f"{len(errors)} errors, {len(warnings)} warnings)")
    return 1 if bad else 0


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def write_manifest(root: str, out_csv: str, with_sha1: bool = True,
                   image_sizes: bool = False) -> int:
    gens = iter_generators(root)
    if not gens:
        print(f"FAIL  no generators found under {root}")
        return 1

    size_fn = None
    if image_sizes:
        try:
            from PIL import Image

            def size_fn(path):  # noqa: F811
                with Image.open(path) as im:
                    return im.width, im.height
        except ImportError:
            print("WARN  Pillow not installed: writing manifest without width/height")

    fields = ["path", "generator", "label", "class", "bytes"]
    if size_fn:
        fields += ["width", "height"]
    if with_sha1:
        fields.append("sha1")

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    rows = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g in gens:
            for side, label in (("real_files", 0), ("fake_files", 1)):
                for p in g[side]:
                    row = {
                        "path": os.path.relpath(p, root).replace(os.sep, "/"),
                        "generator": g["name"],
                        "label": label,
                        "class": "real" if label == 0 else "fake",
                        "bytes": os.path.getsize(p),
                    }
                    if size_fn:
                        wh = size_fn(p)
                        row["width"], row["height"] = wh
                    if with_sha1:
                        row["sha1"] = file_sha1(p)
                    w.writerow(row)
                    rows += 1
                    if rows % 20000 == 0:
                        print(f"  .. {rows} rows")
    print(f"wrote {rows} rows -> {out_csv}")
    stats = Counter()
    for g in gens:
        stats["real"] += len(g["real_files"])
        stats["fake"] += len(g["fake_files"])
    print(f"  {len(gens)} generators, {stats['real']} real / {stats['fake']} fake")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stage and verify AIGC-forensics datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("organize", help="build <generator>/{real,fake} from a download")
    o.add_argument("--src", required=True)
    o.add_argument("--dst", required=True)
    o.add_argument("--layout", choices=["flat", "nested"], default="flat")
    o.add_argument("--mode", choices=["auto", "hardlink", "copy"], default="auto",
                   help="auto = hard-link when possible, else copy (default)")
    o.add_argument("--generator", default=None,
                   help="force one generator name for every pair, e.g. ProGAN")
    o.add_argument("--map", default=None,
                   help='JSON file mapping source dir -> generator name, e.g. '
                        '{"ADM": "ADM", "DiffusionForensics/ADM": "ADM"}')
    o.add_argument("--dry-run", action="store_true")
    o.add_argument("--verbose", action="store_true")

    v = sub.add_parser("verify", help="counts, imbalance, duplicates, leakage")
    v.add_argument("--root", required=True)
    v.add_argument("--leakage-with", default=None,
                   help="second root to compare hashes against (e.g. the train split)")
    v.add_argument("--strict", action="store_true", help="warnings also fail")
    v.add_argument("--expect-real", type=int, default=None)
    v.add_argument("--expect-fake", type=int, default=None)
    v.add_argument("--max-ratio", type=float, default=10.0)
    v.add_argument("--no-hash", action="store_true", help="skip duplicate/leak hashing")

    m = sub.add_parser("manifest", help="CSV inventory of a staged dataset")
    m.add_argument("--root", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--no-sha1", action="store_true")
    m.add_argument("--image-sizes", action="store_true",
                   help="also record width/height (needs Pillow)")

    args = ap.parse_args(argv)

    if args.cmd == "organize":
        mapping = {}
        if args.map:
            with open(args.map, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        organize(args.src, args.dst, layout=args.layout, mode=args.mode,
                 generator=args.generator, mapping=mapping,
                 dry_run=args.dry_run, verbose=args.verbose)
        return 0
    if args.cmd == "verify":
        return verify(args.root, leakage_with=args.leakage_with, strict=args.strict,
                      expect_real=args.expect_real, expect_fake=args.expect_fake,
                      max_ratio=args.max_ratio, no_hash=args.no_hash)
    if args.cmd == "manifest":
        return write_manifest(args.root, args.out, with_sha1=not args.no_sha1,
                              image_sizes=args.image_sizes)
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
