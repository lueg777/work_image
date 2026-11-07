"""Self-checking smoke test. No pytest, no dataset, no GPU needed.

Run with the bundled python (numpy only) to validate everything that can be
validated without torch:

    python scripts/smoke_test.py

Add ``--require-torch`` on a machine with torch to also validate the model
and operators end to end; without it those checks are skipped, not faked.

Every check prints PASS/FAIL with the number it compared, so a failure tells
you what broke rather than just that something did.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.eval import metrics as M            # noqa: E402
from src.eval import spectrum as S           # noqa: E402

FAILED = []
PASSED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"PASS  {name}" + (f"   [{detail}]" if detail else ""))
    else:
        FAILED.append(name)
        print(f"FAIL  {name}" + (f"   [{detail}]" if detail else ""))


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _auc_bruteforce(labels, scores) -> float:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    tot = 0.0
    for p in pos:
        for n in neg:
            tot += 1.0 if p > n else (0.5 if p == n else 0.0)
    return tot / (len(pos) * len(neg))


def _ap_bruteforce(labels, scores) -> float:
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    y = np.asarray(labels)[order]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    hits = 0
    total = 0.0
    for i, yi in enumerate(y, start=1):
        if yi == 1:
            hits += 1
            total += hits / i
    return total / n_pos


def test_metrics() -> None:
    print("\n-- metrics --")
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    # threshold 0.5: predicts 0,0,0,1 -> 3/4 correct
    check("accuracy hand-computed", approx(M.accuracy(labels, scores), 0.75),
          f"{M.accuracy(labels, scores):.4f}")
    c = M.confusion_counts(labels, scores)
    check("confusion counts", c == {"tp": 1, "fp": 0, "tn": 2, "fn": 1}, str(c))
    check("auc perfect separation", approx(M.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0))
    check("auc fully inverted", approx(M.auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0))
    check("auc all ties -> 0.5", approx(M.auc([0, 1, 0, 1], [0.5] * 4), 0.5))

    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.4).astype(int)
    s = rng.random(200) + 0.6 * y
    check("auc == brute force", approx(M.auc(y, s), _auc_bruteforce(y, s), 1e-12),
          f"{M.auc(y, s):.6f} vs {_auc_bruteforce(y, s):.6f}")
    check("ap == brute force", approx(M.average_precision(y, s), _ap_bruteforce(y, s), 1e-12),
          f"{M.average_precision(y, s):.6f} vs {_ap_bruteforce(y, s):.6f}")
    check("ap perfect -> 1.0", approx(M.average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0),
          f"{M.average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]):.4f}")

    tpr = M.accuracy_at_fpr([0, 0, 0, 0, 1, 1, 1, 1],
                            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 0.25)
    check("tpr@fpr25 perfect separation", approx(tpr, 1.0), f"{tpr:.4f}")

    bm = M.binary_metrics(labels, scores)
    check("binary_metrics keys", {"acc", "auc", "ap", "n", "tp", "fp", "tn", "fn"}
          <= set(bm), str(sorted(bm)))
    check("binary_metrics n", bm["n"] == 4.0)


# --------------------------------------------------------------------------- #
# spectrum
# --------------------------------------------------------------------------- #
def _smooth(h=64, w=64) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    g = (yy / h + xx / w) / 2.0
    return np.stack([g] * 3, axis=-1)


def _noise(h=64, w=64, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3))


def test_spectrum() -> None:
    print("\n-- spectrum --")
    check("to_luma uint8 range", approx(S.to_luma(np.full((4, 4, 3), 255, np.uint8))[0, 0], 1.0))
    check("to_luma float passthrough",
          approx(S.to_luma(np.full((4, 4, 3), 0.5))[0, 0], 0.5))

    const = np.full((32, 32, 3), 0.5)
    lp = S.log_power_spectrum(const)
    check("constant image has zero AC power", float(np.abs(lp).max()) < 1e-9,
          f"max={np.abs(lp).max():.2e}")

    noise_ratio = S.hf_energy_ratio(_noise())
    smooth_ratio = S.hf_energy_ratio(_smooth())
    check("noise is more high-frequency than a gradient", noise_ratio > smooth_ratio,
          f"{noise_ratio:.4f} > {smooth_ratio:.4f}")

    prof = S.radial_power_profile(_noise(), n_bins=32)
    check("radial profile length", prof.size == 32, str(prof.size))
    check("radial profile has no NaN", not np.isnan(prof).any())

    # use a coarse binning for the shape checks: with 32 bins the outermost
    # annuli hold only a handful of pixels and are pure sampling noise
    coarse = S.radial_power_profile(_noise(), n_bins=8)
    check("white noise radial profile is flat-ish (max/min < 2)",
          float(coarse.max() / max(coarse.min(), 1e-12)) < 2.0,
          f"ratio={coarse.max()/max(coarse.min(),1e-12):.2f}")

    smooth_prof = S.radial_power_profile(_smooth(), n_bins=8)
    check("gradient energy decays with frequency",
          float(smooth_prof[0]) > float(smooth_prof[-1]) * 100,
          f"{smooth_prof[0]:.3e} > {smooth_prof[-1]:.3e}")
    check("innermost bin is not artificially zero (DC excluded from averaging)",
          float(coarse[0]) > 0 and float(smooth_prof[0]) > 0)

    check("distance to self is 0", approx(S.spectral_distance(prof, prof), 0.0))
    smooth_prof_fine = S.radial_power_profile(_smooth(), n_bins=32)
    d = S.spectral_distance(prof, smooth_prof_fine)
    check("distance is positive and symmetric",
          d > 0 and approx(d, S.spectral_distance(smooth_prof_fine, prof)),
          f"{d:.4f}")

    mat = S.pairwise_spectral_distance([prof, smooth_prof_fine, smooth_prof_fine])
    check("pairwise matrix symmetric with zero diagonal",
          approx(mat[0, 0], 0.0) and approx(mat[0, 1], mat[1, 0])
          and approx(mat[1, 2], 0.0))

    try:
        S.spectral_distance(prof, smooth_prof)
        check("mismatched profile lengths are rejected", False, "no error raised")
    except ValueError:
        check("mismatched profile lengths are rejected", True)

    dv = S.descriptor_vector(_noise())
    check("descriptor_vector keys", set(dv) == {"hf_ratio", "profile_mean",
                                               "profile_std", "profile_peak_bin"},
          str(sorted(dv)))


# --------------------------------------------------------------------------- #
# operators + model (torch)
# --------------------------------------------------------------------------- #
def test_ops_and_model() -> None:
    print("\n-- operators + model (torch) --")
    try:
        import torch
    except ImportError:
        print("SKIP  torch not installed: operator and model checks not run")
        return

    from src.ops.residual_ops import REGISTRY, make_residual_op
    from src.models.npr_resnet import NPRResNet, count_parameters

    x = torch.rand(2, 3, 64, 64)
    for name in REGISTRY:
        op = make_residual_op(name)
        y = op(x)
        ok = y.shape == x.shape and torch.isfinite(y).all()
        check(f"op[{name}] shape+finite", bool(ok), f"scale={op.scale}")

    # the official NPR operator must equal x - up_nearest(down_nearest(x)), *2/3
    import torch.nn.functional as F
    op = make_residual_op("npr_nearest")
    ref = F.interpolate(F.interpolate(x, scale_factor=0.5, mode="nearest",
                                      recompute_scale_factor=True),
                        scale_factor=2.0, mode="nearest",
                        recompute_scale_factor=True)
    ref = (x - ref) * 2.0 / 3.0
    check("npr_nearest matches official formula exactly",
          torch.allclose(op(x), ref, atol=1e-6),
          f"max|d|={float((op(x)-ref).abs().max()):.2e}")

    # a constant image has no high-frequency content: every high-pass op -> 0
    const = torch.full((1, 3, 32, 32), 0.5)
    for name in ("npr_nearest", "npr_bilinear", "npr_bicubic", "npr_box",
                 "laplacian", "dct_lowpass"):
        v = float(make_residual_op(name)(const).abs().max())
        check(f"op[{name}] annihilates a constant image", v < 1e-5, f"max={v:.2e}")

    ident = make_residual_op("identity")(x)
    check("identity returns the input unchanged", torch.allclose(ident, x))

    model = NPRResNet(layers=(3, 4), num_classes=1, op="npr_nearest")
    n = count_parameters(model)
    check("param count is in the NPR ballpark (1.3M-1.6M)", 1.3e6 < n < 1.6e6,
          f"{n:,}")
    out = model(torch.rand(2, 3, 224, 224))
    check("forward gives (B, 1) logits", tuple(out.shape) == (2, 1), str(tuple(out.shape)))
    check("fc1 input is 512 features", model.fc1.in_features == 512,
          str(model.fc1.in_features))
    check("no layer3/layer4 (NPR head reads layer2)",
          not hasattr(model, "layer3") and not hasattr(model, "layer4"))

    # one optimisation step must actually move the loss down
    y = torch.tensor([0.0, 1.0]).reshape(-1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()
    xb = torch.rand(2, 3, 224, 224)
    losses = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(xb), y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    check("three Adam steps reduce the loss", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")

    # a state_dict must not carry operator buffers (else NPR.pth mismatches)
    sd = model.state_dict()
    check("state_dict has no op buffers",
          not any(k.startswith("op.") for k in sd), str([k for k in sd
                                                        if k.startswith("op.")][:4]))


def test_data_prep() -> None:
    """Stage / verify / manifest against a throw-away tree of fake images.

    The images are random bytes with an image extension: nothing here opens
    them, which is the point -- dataset preparation must work on a box with no
    torch and no Pillow.
    """
    print("\n-- data prep --")
    import importlib.util
    import shutil

    from src.layout import classify_class_dir, iter_image_files, pair_class_dirs

    check("classify 0_real -> real", classify_class_dir("0_real") == "real")
    check("classify 1-fake -> fake", classify_class_dir("1-Fake") == "fake")
    check("generator name ProGAN reads as fake-like on its own",
          classify_class_dir("ProGAN") == "fake")
    check("pair finds the cnndetection convention",
          pair_class_dirs(["0_real", "1_fake"]) == ("0_real", "1_fake", True))
    check("pair finds real/fake", pair_class_dirs(["fake", "real"]) == ("real", "fake", True))
    check("pair refuses two generator-shaped names",
          pair_class_dirs(["ProGAN", "StyleGAN"]) is None)
    check("pair guesses an unlabelled folder only next to a recognisable one",
          pair_class_dirs(["imgs", "fake"]) == ("imgs", "fake", False))

    spec = importlib.util.spec_from_file_location(
        "prepare_data", os.path.join(REPO_ROOT, "scripts", "prepare_data.py"))
    pd = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    spec.loader.exec_module(pd)                          # type: ignore[union-attr]

    # A scratch dir inside the repo, not tempfile.mkdtemp(): under the DSH
    # sandbox a 0o700 temp dir gets a DACL that denies our own writes.
    scratch = os.path.join(REPO_ROOT, "_smoke_scratch")
    shutil.rmtree(scratch, ignore_errors=True)
    src = os.path.join(scratch, "src")

    def make(rel: str, n: int) -> None:
        d = os.path.join(src, *rel.split("/"))
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            name = rel.replace("/", "_") + f"_{i}.png"
            with open(os.path.join(d, name), "wb") as f:
                f.write(os.urandom(64))

    make("GenA/0_real", 3)
    make("GenA/1_fake", 3)
    make("GenB/real", 2)
    make("GenB/fake", 2)

    dst = os.path.join(scratch, "dst")
    first = pd.organize(src, dst)
    check("organize discovers both generators",
          sorted(first["generators"]) == ["GenA", "GenB"], str(first["generators"]))
    check("organize counts all 10 images", first["stats"].get("files") == 10,
          str(first["stats"]))
    check("staged layout is <gen>/real|fake",
          len(iter_image_files(os.path.join(dst, "GenA", "real"))) == 3
          and len(iter_image_files(os.path.join(dst, "GenA", "fake"))) == 3)

    try:
        import torch  # noqa: F401
        from src.data.datasets import BinaryForensicsDataset
    except Exception:
        print("SKIP  loader check needs torch")
    else:
        # construction never opens files, so random bytes are fine here: this
        # only proves prepare_data stages a tree the loader will accept
        staged = BinaryForensicsDataset(os.path.join(dst, "GenA"))
        check("loader accepts what prepare_data staged, labels pinned",
              staged.class_to_idx == {"real": 0, "fake": 1}
              and sum(1 for _, t in staged.samples if t == 0) == 3
              and sum(1 for _, t in staged.samples if t == 1) == 3,
              str(staged.class_to_idx))
        check("loader sees 6 samples for GenA", len(staged) == 6, str(len(staged)))

    second = pd.organize(src, dst)
    check("re-running organize skips identical files instead of duplicating",
          second["stats"].get("skipped") == 10, str(second["stats"]))

    # auto mode hard-links where it can, so a staged file *is* the source file.
    # That is the point (no second copy of 30GB), but it means editing a staged
    # file edits the download. Assert the aliasing so the trade-off stays visible.
    probe = os.path.join(dst, "GenA", "real", "GenA_0_real_0.png")
    original = os.path.join(src, "GenA", "0_real", "GenA_0_real_0.png")
    check("auto mode hard-links instead of copying (staged file is the source inode)",
          os.path.samefile(probe, original))

    # Break the link the way a user would (replace the staged file), then make
    # its content differ from the source so the re-run must rename, not skip.
    clash = probe
    os.remove(clash)
    with open(clash, "wb") as f:
        f.write(b"different content")
    third = pd.organize(src, dst)
    check("a name collision with different content is renamed, never overwritten",
          third["stats"].get("renamed") == 1
          and len(iter_image_files(os.path.dirname(clash))) == 4,
          str(third["stats"]))

    check("verify passes on a clean split", pd.verify(dst) == 0)

    out_csv = os.path.join(scratch, "manifest.csv")
    pd.write_manifest(dst, out_csv)
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    check("manifest row count matches the staged images", len(rows) == 11,
          str(len(rows)))
    check("manifest pins real=0 / fake=1",
          all((r["label"] == "0") == (r["class"] == "real") for r in rows)
          and {r["generator"] for r in rows} == {"GenA", "GenB"})
    check("manifest carries a sha1 per row",
          all(len(r["sha1"]) == 40 for r in rows))

    check("verify flags leakage when the same image is on both sides",
          pd.verify(dst, leakage_with=dst) == 1)

    broken = os.path.join(dst, "Broken")
    os.makedirs(os.path.join(broken, "real"), exist_ok=True)
    os.makedirs(os.path.join(broken, "fake"), exist_ok=True)
    with open(os.path.join(broken, "fake", "only.png"), "wb") as f:
        f.write(b"only a fake side")
    check("verify fails on an empty real side", pd.verify(dst) == 1)

    shutil.rmtree(scratch, ignore_errors=True)


def test_transforms() -> None:
    print("\n-- transforms --")
    try:
        from PIL import Image
        from src.data.transforms import JpegCompress, DownUpResize
    except ImportError:
        print("SKIP  torchvision/PIL not installed")
        return
    img = Image.fromarray((np.random.default_rng(1).random((40, 50, 3)) * 255)
                          .astype(np.uint8))
    j = JpegCompress(50)(img)
    check("jpeg keeps size and mode", j.size == img.size and j.mode == "RGB")
    d = DownUpResize(0.5)(img)
    check("down/up resize restores size", d.size == img.size, str(d.size))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-torch", action="store_true",
                    help="fail instead of skipping when torch is missing")
    args = ap.parse_args()

    for fn in (test_metrics, test_spectrum, test_data_prep,
               test_transforms, test_ops_and_model):
        try:
            fn()
        except Exception:
            FAILED.append(fn.__name__ + " (raised)")
            print(f"FAIL  {fn.__name__} raised:")
            traceback.print_exc()

    if args.require_torch:
        try:
            import torch  # noqa: F401
        except ImportError:
            FAILED.append("--require-torch but torch missing")

    print(f"\n=== {len(PASSED)} passed, {len(FAILED)} failed ===")
    if FAILED:
        for f in FAILED:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
