"""Preprocessing that is bit-faithful to the official NPR pipeline.

Verified against `chuangchuangtan/NPR-DeepfakeDetection/data/datasets.py`
(fetched 2026-10-31). The official `binary_dataset` composes, in this order:

    rz_func   = Resize((loadSize, loadSize))          # skipped if test & no_resize
    crop_func = RandomCrop(cropSize) if train
                identity              if no_crop
                CenterCrop(cropSize)  otherwise
    flip_func = RandomHorizontalFlip() if train and not no_flip else identity
    ToTensor()
    Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

Two details that are easy to get wrong and that silently wreck comparability
with the released `NPR.pth` checkpoint:

*   It is ``RandomCrop``, **not** ``RandomResizedCrop``. The crop never
    rescales, so the pixel grid statistics the network sees stay at the
    original resolution scale.
*   ``Resize`` is a plain square resize to ``(loadSize, loadSize)``
    (default 256), then a 224 crop. Test time is resize + centre crop, i.e.
    the official defaults are ``no_resize=False, no_crop=False``.

The ``translate_duplicate`` variant used for the GenImage benchmark replaces
the resize with a reflection-free tiling, so that the generator's native
sampling grid is preserved instead of being resampled. That behaviour is
reproduced in `TranslateDuplicate` and is marked EXPERIMENTAL because its
exact reference implementation was reconstructed from the README rather than
copied.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "TranslateDuplicate",
    "JpegCompress",
    "DownUpResize",
    "build_train_transform",
    "build_eval_transform",
    "build_transform",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# GenImage-style tiling (EXPERIMENTAL)
# --------------------------------------------------------------------------- #
class TranslateDuplicate:
    """Tile the image by integer translation until it covers ``size``, then crop.

    Instead of interpolating (which destroys the high-frequency fingerprint we
    are trying to measure), the image is duplicated along both axes with
    ``math.ceil`` repetition and the centre ``size x size`` window is taken.
    """

    def __init__(self, size: int = 224):
        self.size = int(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        nx = -(-self.size // max(w, 1)) + 1
        ny = -(-self.size // max(h, 1)) + 1
        canvas = Image.new(img.mode, (w * nx, h * ny))
        for i in range(nx):
            for j in range(ny):
                canvas.paste(img, (i * w, j * h))
        left = (canvas.size[0] - self.size) // 2
        top = (canvas.size[1] - self.size) // 2
        return canvas.crop((left, top, left + self.size, top + self.size))


# --------------------------------------------------------------------------- #
# degradations used by the robustness sweep
# --------------------------------------------------------------------------- #
class JpegCompress:
    """Round-trip through JPEG at a fixed quality (applied on the PIL image)."""

    def __init__(self, quality: int):
        self.quality = int(quality)

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out.load()
        buf.close()
        return out


class DownUpResize:
    """Downscale by ``factor`` then upscale back: simulates re-sharing at low res.

    Uses bicubic in both directions, matching the common convention for this
    kind of attack. The output size equals the input size, so a subsequent
    crop still lines up.
    """

    def __init__(self, factor: float = 0.5):
        if not 0 < factor <= 1:
            raise ValueError("factor must be in (0, 1]")
        self.factor = float(factor)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        small = (max(1, int(round(w * self.factor))),
                 max(1, int(round(h * self.factor))))
        down = img.resize(small, Image.BICUBIC)
        return down.resize((w, h), Image.BICUBIC)


def _degradation(spec: Optional[dict]):
    """Turn {'jpeg': 75} / {'resize': 0.5} into a callable, or None."""
    if not spec:
        return None
    kind = (spec.get("kind") or "").lower()
    if kind in ("jpeg", "jpg"):
        return JpegCompress(int(spec["quality"]))
    if kind in ("resize", "downup"):
        return DownUpResize(float(spec["factor"]))
    raise ValueError(f"unknown degradation kind: {spec.get('kind')!r}")


# --------------------------------------------------------------------------- #
# public builders
# --------------------------------------------------------------------------- #
def build_transform(
    crop_size: int = 224,
    load_size: int = 256,
    train: bool = False,
    no_crop: bool = False,
    no_resize: bool = False,
    flip: bool = True,
    resize_mode: str = "resize",
    degradation: Optional[dict] = None,
    degradation_stage: str = "after_resize",
) -> transforms.Compose:
    """Reproduce the official pipeline, with the ablation knobs exposed.

    Parameters
    ----------
    crop_size, load_size
        Official defaults are 224 / 256.
    train
        Selects RandomCrop + flip instead of CentreCrop.
    no_crop, no_resize
        Official test-time escape hatches; both default False.
    resize_mode
        ``"resize"`` (official) or ``"translate_duplicate"`` (GenImage variant).
    degradation
        Optional post-processing attack, e.g. ``{"kind": "jpeg", "quality": 75}``
        or ``{"kind": "resize", "factor": 0.5}``.
    degradation_stage
        ``"after_resize"`` (default, matches CNNDetection's composition order:
        resize -> augment -> crop) or ``"before_resize"``.
    """
    ops = []

    degrade = _degradation(degradation)
    if degrade is not None and degradation_stage == "before_resize":
        ops.append(degrade)

    if not (no_resize and not train):
        if resize_mode == "translate_duplicate":
            ops.append(TranslateDuplicate(crop_size if no_crop else load_size))
        elif resize_mode == "resize":
            ops.append(transforms.Resize((load_size, load_size)))
        else:
            raise ValueError(f"unknown resize_mode: {resize_mode!r}")

    if degrade is not None and degradation_stage == "after_resize":
        ops.append(degrade)

    if train:
        ops.append(transforms.RandomCrop(crop_size))
    elif no_crop:
        pass
    else:
        ops.append(transforms.CenterCrop(crop_size))

    if flip and train:
        ops.append(transforms.RandomHorizontalFlip())

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)


def build_train_transform(cfg: dict) -> transforms.Compose:
    d = cfg.get("data", {})
    return build_transform(
        crop_size=int(d.get("crop_size", 224)),
        load_size=int(d.get("load_size", 256)),
        train=True,
        flip=bool(d.get("flip", True)),
        resize_mode=d.get("resize_mode", "resize"),
    )


def build_eval_transform(cfg: dict, degradation: Optional[dict] = None) -> transforms.Compose:
    d = cfg.get("data", {})
    return build_transform(
        crop_size=int(d.get("crop_size", 224)),
        load_size=int(d.get("load_size", 256)),
        train=False,
        no_crop=bool(d.get("eval_no_crop", False)),
        no_resize=bool(d.get("eval_no_resize", False)),
        resize_mode=d.get("resize_mode", "resize"),
        degradation=degradation,
        degradation_stage=d.get("degradation_stage", "after_resize"),
    )
