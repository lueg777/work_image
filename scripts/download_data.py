"""Fetch the public AIGC-forensics datasets and stage them for ``src/``.

Why a separate script: the official packages are *not* laid out the way the
loader wants them (``0_real`` / ``1_fake``, an extra benchmark level, and --
for ProGAN / StyleGAN / StyleGAN2 / CycleGAN -- one more level of object
classes inside a single generator).  Doing that by hand is how label flips and
duplicated "generators" get in.  So this script downloads (or accepts an
already-downloaded archive), extracts, stages into ``<gen>/{real,fake}`` and
then runs the same checker the loader uses.

Data sources (verified against the official READMEs, 2026-11):

  AIGCDetectBenchmark  16 generators, the numbers in the NPR paper's big table
                       https://drive.google.com/drive/folders/1p4ewuAo7d5LbNJ4cKyh10Xl9Fg2yoFOw
  ForenSynths (train)  CNNDetection's 20 ProGAN classes, we take 4
                       https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b
  GANGen-Detection     https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj
  DiffusionForensics   https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf
  UniversalFakeDetect  https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-
  Diffusion1kStep      https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq

Google Drive is not reachable from every network (it times out behind the
usual corporate/edu firewall).  When that happens use ``--local-archive`` or
``--local-dir`` with something you downloaded by hand; the staging, the
multiclass merge and the verification are identical either way.

Examples
--------
    python scripts/download_data.py --check
    python scripts/download_data.py --dataset benchmark
    python scripts/download_data.py --dataset benchmark --dest <dl>
    python scripts/download_data.py --dataset benchmark --local-dir <dl>/AIGCDetectBenchmark
    python scripts/download_data.py --dataset forensynths --local-dir <dl>/ForenSynths/train

``<dl>`` is a download root you can write to. It defaults to
``~/aigc_data`` (i.e. ``%USERPROFILE%\\aigc_data`` on Windows) and is created
if missing; 30 GB must be free there. Keep it on the same volume as
``--into`` (default ``<repo>/datasets``) so staging uses hard links and does
not cost a second 30 GB.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tarfile
import zipfile
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")

# Where the raw download lands.  Home rather than a fixed "D:/dl": most Windows
# machines have no D: at all, and %USERPROFILE% is always writable without
# elevation while C:\ itself usually is not.
DEFAULT_DEST = os.path.join(os.path.expanduser("~"), "aigc_data")

# Generator names the two benchmark suites ship (NPR / CNNDetection READMEs).
# Used only to decide *which directory level is the generator* when a folder
# contains several real/fake pairs -- not to classify images.
KNOWN_GENERATORS = {
    "progan", "stylegan", "stylegan2", "stylegan3", "biggan", "cyclegan",
    "stargan", "gaugan", "wfir", "sanitized",
    "adm", "guided", "guideddiffusion", "improveddiffusion", "iddpm", "ddpm",
    "glide", "midjourney", "sdv14", "sdv15", "sdv21", "sdv4", "sdv5", "sdxl",
    "vqdm", "vqdiffusion", "wukong", "dalle2",
}


def _norm(name: str) -> str:
    """`SDv1.4` / `sd-v1.4` / `SDv1_4` all collapse to `sdv14`."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


GDRIVE = {
    "benchmark": ("AIGCDetectBenchmark",
                  "https://drive.google.com/drive/folders/1p4ewuAo7d5LbNJ4cKyh10Xl9Fg2yoFOw"),
    "gengen": ("GANGen-Detection",
               "https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj"),
    "diffusionforensics": ("DiffusionForensics",
                           "https://drive.google.com/drive/folders/1jZE4hg6SxRvKaPYO_yyMeJN_DOcqGMEf"),
    "universalfakedetect": ("UniversalFakeDetect",
                            "https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-"),
    "diffusion1kstep": ("Diffusion1kStep",
                        "https://drive.google.com/drive/folders/14f0vApTLiukiPvIHukHDzLujrvJpDpRq"),
}

BAIDU_FORENSYNTHS = "https://pan.baidu.com/s/1l-rXoVhoc8xJDl20Cdwy4Q?pwd=ft8b"
PROBE_HOSTS = ("drive.google.com", "pan.baidu.com")

# The backup packages name the same generator differently.  Only aliases stated
# outright in the official READMEs are mapped -- guessing the rest is how one
# results table ends up with two rows for one model.
NAME_ALIASES = {
    "guided": "ADM",                # NPR README: "guided # Also known as ADM"
    "guided-diffusion": "ADM",      # Diffusion1kStep
    "guided_diffusion": "ADM",
    "improved-diffusion": "IDDPM",  # Diffusion1kStep
}

# what the loader will accept as one generator's two sides
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402


def _load_prepare_data():
    path = os.path.join(SCRIPTS, "prepare_data.py")
    spec = spec_from_file_location("prepare_data", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pd = _load_prepare_data()


# --------------------------------------------------------------------------- #
# environment checks
# --------------------------------------------------------------------------- #
def probe(host: str, timeout: float = 6.0) -> str:
    """'ok', 'timeout' or the exception name -- no request, just a TLS connect."""
    try:
        sock = socket.create_connection((host, 443), timeout=timeout)
    except socket.timeout:
        return "timeout"
    except OSError as exc:
        return type(exc).__name__
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(sock, server_hostname=host) as tls:
            return f"ok ({tls.version()})"
    except Exception as exc:  # noqa: BLE001 - report, never raise from a probe
        return type(exc).__name__


def have(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def report_environment() -> Dict[str, str]:
    state = {f"net:{h}": probe(h) for h in PROBE_HOSTS}
    state["tool:gdown"] = have("gdown") or "missing"
    state["tool:7z"] = have("7z") or have("7za") or "missing"
    return state


# --------------------------------------------------------------------------- #
# download / extract
# --------------------------------------------------------------------------- #
def download_gdrive(name: str, dest: str) -> str:
    label, url = GDRIVE[name]
    target = os.path.join(dest, label)
    if os.path.isdir(target) and any(os.scandir(target)):
        print(f"[skip] {target} already has content")
        return target
    if probe("drive.google.com") != "ok" and not probe("drive.google.com").startswith("ok"):
        raise SystemExit(
            "drive.google.com is not reachable from this machine.\n"
            f"Download it by hand ({url}) and re-run with --local-dir/--local-archive."
        )
    gdown = have("gdown")
    if not gdown:
        raise SystemExit(
            "gdown is not installed. Either\n"
            "  pip install gdown==4.7.1\n"
            f"or download {url} by hand and use --local-dir/--local-archive."
        )
    os.makedirs(target, exist_ok=True)
    cmd = [gdown, "--folder", url, "--output", target, "--remaining-ok"]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=False)
    if not any(os.scandir(target)):
        raise SystemExit(f"gdown produced nothing in {target}")
    return target


def extract_archives(root: str, verbose: bool = True) -> List[str]:
    """Extract every archive under ``root`` next to itself. Returns the dirs."""
    made: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            low = name.lower()
            out = os.path.join(dirpath, os.path.splitext(name)[0])
            if low.endswith(".zip"):
                os.makedirs(out, exist_ok=True)
                if verbose:
                    print(f"  unzip {name}")
                with zipfile.ZipFile(path) as zf:
                    zf.extractall(out)
                made.append(out)
            elif low.endswith((".tar", ".tar.gz", ".tgz")):
                os.makedirs(out, exist_ok=True)
                if verbose:
                    print(f"  untar {name}")
                with tarfile.open(path) as tf:
                    tf.extractall(out)
                made.append(out)
            elif low.endswith((".7z", ".rar")):
                seven = have("7z") or have("7za")
                if not seven:
                    print(f"  [skip] {name}: need 7z (apt install p7zip-full / 7-Zip on Windows)")
                    continue
                os.makedirs(out, exist_ok=True)
                if verbose:
                    print(f"  unpack {name}")
                subprocess.run([seven, "x", "-y", f"-o{out}", path], check=False)
                made.append(out)
    return made


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #
def plan_generators(pairs: List[Dict], root: str) -> Dict[str, List[Dict]]:
    """Group real/fake pairs into generators, merging multiclass generators.

    CNNDetection ships ProGAN / StyleGAN / StyleGAN2 / CycleGAN as
    ``<gen>/<object class>/{0_real,1_fake}``.  Taken literally that yields 20
    "generators" named after objects -- the exact trap the loader warns about.
    A pair whose sibling directories are also pairs is a class of the parent.
    """
    by_parent: Dict[str, List[Dict]] = {}
    for pair in pairs:
        by_parent.setdefault(os.path.dirname(pair["gen_dir"]), []).append(pair)

    groups: Dict[str, List[Dict]] = {}
    for parent, siblings in by_parent.items():
        parent_name = _norm(os.path.basename(parent))
        child_names = {_norm(os.path.basename(p["gen_dir"])) for p in siblings}
        merge = False
        if len(siblings) > 1 and os.path.abspath(parent) != os.path.abspath(root):
            if parent_name in KNOWN_GENERATORS:
                # <gen>/<object class>/{0_real,1_fake}: the classes belong together
                merge = True
            elif child_names & KNOWN_GENERATORS:
                # <benchmark>/<gen>/{real,fake}: the siblings *are* generators,
                # fusing them would report one bogus "AIGCDetectBenchmark" row
                merge = False
            else:
                merge = True
        if merge:
            name = os.path.basename(parent)
            groups.setdefault(name, []).extend(siblings)
        else:
            for pair in siblings:
                groups.setdefault(os.path.basename(pair["gen_dir"]), []).append(pair)
    return groups


def stage(groups: Dict[str, List[Dict]], dst: str, mode: str = "auto",
          dry_run: bool = False, verbose: bool = False,
          alias: bool = True) -> Dict[str, int]:
    stats = {"files": 0, "linked": 0, "copied": 0, "skipped": 0, "renamed": 0}
    for gen in sorted(groups):
        out_name = NAME_ALIASES.get(gen.lower(), gen) if alias else gen
        if out_name != gen:
            print(f"  alias: {gen} -> {out_name}")
        merged = len(groups[gen]) > 1
        for pair in groups[gen]:
            prefix = os.path.basename(pair["gen_dir"]) if merged else ""
            for side in ("real", "fake"):
                files = pd.iter_image_files(pair[side])
                if not files:
                    print(f"  [warn] {pair[side]}: no images")
                    continue
                out_dir = os.path.join(dst, out_name, side)
                if not dry_run:
                    os.makedirs(out_dir, exist_ok=True)
                for path in files:
                    base = os.path.basename(path)
                    if prefix:
                        base = f"{prefix}__{base}"
                    target = os.path.join(out_dir, base)
                    stats["files"] += 1
                    if os.path.exists(target):
                        if pd._same_content(target, path):
                            stats["skipped"] += 1
                            continue
                        target = pd._unique_target(out_dir, base, set())
                        stats["renamed"] += 1
                    if dry_run:
                        continue
                    action = pd._place(path, target, mode)
                    stats[action] += 1
                    if verbose:
                        print(f"    {action}: {target}")
    return stats


# --------------------------------------------------------------------------- #
# per-dataset run
# --------------------------------------------------------------------------- #
def run_benchmark(src: str, into: str, mode: str, dry_run: bool, verbose: bool,
                  alias: bool = True) -> int:
    pairs = pd.collect_pairs(src)
    if not pairs:
        raise SystemExit(
            f"no <generator>/{{real,fake}} pairs under {src}\n"
            "expected 0_real/1_fake, real/fake, or <gen>/<class>/{0_real,1_fake}"
        )
    groups = plan_generators(pairs, src)
    print(f"\nfound {len(pairs)} real/fake pair(s) -> {len(groups)} generator(s):")
    for gen in sorted(groups):
        n = sum(len(pd.iter_image_files(p[s])) for p in groups[gen] for s in ("real", "fake"))
        extra = f"  (merged from {len(groups[gen])} class dirs)" if len(groups[gen]) > 1 else ""
        print(f"  {gen:<22} {n:>6} images{extra}")

    dst = os.path.join(into, "Generalization_Test") if into else "Generalization_Test"
    stats = stage(groups, dst, mode=mode, dry_run=dry_run, verbose=verbose, alias=alias)
    print(f"\nstaged into {dst}: {stats}")
    if dry_run:
        print("dry run: nothing written")
        return 0
    return pd.verify(dst)


def run_forensynths(src: str, into: str, mode: str, dry_run: bool, verbose: bool,
                    split: str = "train") -> int:
    # the loader wants ForenSynths_train_val/<split>/<class>/{real,fake}; the
    # official zip is <class>/{0_real,1_fake} per split, so the split has to be
    # inferred (or passed) rather than guessed from the class names
    if split == "auto":
        split = os.path.basename(os.path.normpath(src)).lower()
        if split not in ("train", "val", "test"):
            raise SystemExit(f"cannot infer split from {src}; pass --split train|val")
    pairs = pd.collect_pairs(src)
    if not pairs:
        raise SystemExit(f"no <class>/{{real,fake}} pairs under {src}")
    names = {os.path.basename(p["gen_dir"]) for p in pairs}
    wanted = pd.FORENSYNTHS_CLASSES
    keep = [p for p in pairs if os.path.basename(p["gen_dir"]).lower() in wanted]
    if len(keep) != len(pairs):
        print(f"keeping only {sorted(wanted & {n.lower() for n in names})} "
              f"({len(pairs) - len(keep)} other classes ignored)")
    if not keep:
        raise SystemExit(f"none of the 4 classes {sorted(wanted)} found; saw {sorted(names)}")
    dst = os.path.join(into, "ForenSynths_train_val", split) if into else \
        os.path.join("ForenSynths_train_val", split)
    stats = stage({os.path.basename(p["gen_dir"]): [p] for p in keep}, dst,
                  mode=mode, dry_run=dry_run, verbose=verbose)
    print(f"\nstaged into {dst}: {stats}")
    if dry_run:
        return 0
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="benchmark",
                    choices=["benchmark", "forensynths"] + list(GDRIVE),
                    help="which package to fetch (default: benchmark = the 16 generators)")
    ap.add_argument("--dest", default=DEFAULT_DEST,
                    help=f"download root (default: {DEFAULT_DEST})")
    ap.add_argument("--into", default=os.path.join(REPO_ROOT, "datasets"),
                    help="where the staged datasets/ tree goes")
    ap.add_argument("--local-dir", default=None,
                    help="skip downloading, use this already-extracted directory")
    ap.add_argument("--local-archive", default=None,
                    help="skip downloading, extract this archive first")
    ap.add_argument("--mode", default="auto", choices=["auto", "hardlink", "copy"],
                    help="hardlink saves ~30GB but rewrites leak into the original")
    ap.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"],
                    help="ForenSynths split to stage (default: infer from --local-dir name)")
    ap.add_argument("--no-alias", action="store_true",
                    help="keep generator names exactly as shipped (guided, improved-diffusion, ...)")
    ap.add_argument("--check", action="store_true", help="only report network/tools")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.check:
        state = report_environment()
        for key, value in state.items():
            print(f"  {key:<26} {value}")
        reach = state.get("net:drive.google.com", "")
        if not reach.startswith("ok"):
            print("\nGoogle Drive is unreachable here -> download by hand and use "
                  "--local-dir/--local-archive (see the module docstring for links).")
        return 0

    if args.local_archive:
        root = os.path.splitext(os.path.join(args.dest, os.path.basename(args.local_archive)))[0]
        os.makedirs(root, exist_ok=True)
        shutil.copy2(args.local_archive, root)
        extract_archives(root)
        src = next((os.path.join(root, d) for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d))), root)
    elif args.local_dir:
        src = args.local_dir
    else:
        if args.dataset == "forensynths":
            raise SystemExit(
                "ForenSynths lives on Baidu Pan, which needs a browser login and "
                "cannot be scripted reliably.\n"
                f"  1) open {BAIDU_FORENSYNTHS}  (code ft8b) and download\n"
                "  2) re-run:  python scripts/download_data.py --dataset forensynths "
                "--local-dir <extracted>/train"
            )
        src = download_gdrive(args.dataset, args.dest)
        extract_archives(src)

    if not os.path.isdir(src):
        raise SystemExit(f"not a directory: {src}")
    if args.dataset == "forensynths":
        return run_forensynths(src, args.into, args.mode, args.dry_run, args.verbose, args.split)
    return run_benchmark(src, args.into, args.mode, args.dry_run, args.verbose,
                           alias=not args.no_alias)


if __name__ == "__main__":
    sys.exit(main())
