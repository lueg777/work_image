"""Torch-free, PIL-free helpers describing the on-disk dataset layout.

Deliberately dependency-free so that ``scripts/prepare_data.py`` can stage and
verify datasets on a bare python, before the deep-learning environment exists.
``src/data/datasets.py`` imports the *same* functions, so the loader and the
data tooling can never disagree about what counts as a ``real``/``fake`` pair.

Contract (the one both CNNDetection and NPR ship)::

    <root>/<generator>/real/<image>
    <root>/<generator>/fake/<image>

Class folder names differ in the wild, and the differences are load-bearing::

    0_real | real | natural | authentic      -> real  (label 0)
    1_fake | fake | synthetic | ai | generated -> fake (label 1)

``torchvision.datasets.ImageFolder`` sorts class folders alphabetically, which
turns ``fake`` into class 0 and ``real`` into class 1 -- every AUC sign flips
and nothing raises. Labels here are therefore derived from *names*, never from
sort order.

Generators are recognised by the *pairing* rule, not by a name list: a
directory is a generator directory only when exactly one of its subdirectories
looks real-like and exactly one looks fake-like. That keeps a generator called
``ProGAN`` (whose normalised name is in :data:`TRUTHY_FAKE`) from being
mistaken for a class folder.
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, List, Optional, Tuple

__all__ = [
    "IMAGE_EXTS",
    "TRUTHY_FAKE",
    "TRUTHY_REAL",
    "normalize_class_name",
    "classify_class_dir",
    "pair_class_dirs",
    "is_image_file",
    "iter_image_files",
    "file_sha1",
]

IMAGE_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm",
})

TRUTHY_FAKE = frozenset({"fake", "1", "synthetic", "ai", "generated", "stylegan", "progan"})
TRUTHY_REAL = frozenset({"real", "0", "natural", "authentic", "genuine"})


def normalize_class_name(name: str) -> str:
    """``0_real`` -> ``real``; ``1-Fake`` -> ``fake``; ``fake_imgs`` -> kept."""
    s = name.lower().replace("-", "_").replace(" ", "_")
    return s.strip("0123456789_")


def classify_class_dir(name: str) -> Optional[str]:
    """``"real"`` / ``"fake"`` / ``None`` for one directory *name*."""
    n = normalize_class_name(name)
    if not n:
        return None
    if n in TRUTHY_REAL:
        return "real"
    if n in TRUTHY_FAKE:
        return "fake"
    if "real" in n:
        return "real"
    if "fake" in n:
        return "fake"
    return None


def pair_class_dirs(names: Iterable[str]) -> Optional[Tuple[str, str, bool]]:
    """Find the real/fake subdirectory pair among ``names``.

    Returns ``(real_name, fake_name, exact)`` or ``None``.

    ``exact`` is ``True`` when both sides were recognised by their names; the
    ``False`` fallback (exactly two folders, one of which looks like the right
    side) exists for hand-made folders and should be reported to the user,
    because guessing here is how label flips get introduced.
    """
    names = list(names)
    real = [n for n in names if classify_class_dir(n) == "real"]
    fake = [n for n in names if classify_class_dir(n) == "fake"]
    if len(real) == 1 and len(fake) == 1 and real[0] != fake[0]:
        return real[0], fake[0], True
    if len(names) == 2:
        # Fallback for hand-made folders: exactly one recognisable side and one
        # unlabelled side. Deliberately refuses when *both* names look the same
        # way (two generator names such as ProGAN/StyleGAN both read as "fake"),
        # because guessing there is how a whole split ends up on the wrong side.
        first, second = names
        k1, k2 = classify_class_dir(first), classify_class_dir(second)
        if (k1, k2) == (None, "fake"):
            return first, second, False
        if (k1, k2) == ("fake", None):
            return second, first, False
        if (k1, k2) == ("real", None):
            return first, second, False
        if (k1, k2) == (None, "real"):
            return second, first, False
    return None


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def iter_image_files(root: str) -> List[str]:
    """Sorted image files directly under ``root`` (not recursive)."""
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, n) for n in os.listdir(root)
                  if is_image_file(n) and os.path.isfile(os.path.join(root, n)))


def file_sha1(path: str, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-1 -- identical images on different paths hash identically."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
