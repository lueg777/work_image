"""Dataset plumbing for the ForenSynths / Generalization_Test layout.

Expected on-disk layout (the one both CNNDetection and NPR use)::

    datasets/
      ForenSynths_train_val/
        train/{car,cat,chair,horse}/          <- 4 generator classes, flat
        val/{car,cat,chair,horse}/
      Generalization_Test/
        ForenSynths_test/ProGAN/{real,fake}/
        DiffusionForensics/ADM/{real,fake}/
        GANGen-Detection/Glide/{real,fake}/
        ...
        Diffusion1kStep/DALLE2/{real,fake}/

Two label conventions matter and they disagree in the wild:

*   The **training** split is not a real/fake split. Every image under
    ``train/<generator>/`` is a *synthetic* image; the "real" side of training
    comes from the ProGAN source images and is folded in by class label. Our
    training path therefore just uses plain ``ImageFolder`` semantics and the
    class index is the generator identity -- do not interpret it as real/fake.
*   The **test** splits are real/fake. ``ImageFolder`` sorts class folders
    alphabetically, which yields ``fake -> 0, real -> 1``. That silently flips
    every AUC sign and every "probability of being fake" downstream, so
    `BinaryForensicsDataset` pins the mapping to ``real -> 0, fake -> 1``
    explicitly instead of relying on sort order.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..layout import TRUTHY_FAKE, TRUTHY_REAL, pair_class_dirs

__all__ = [
    "BinaryForensicsDataset",
    "list_generators",
    "describe_split",
    "TRUTHY_FAKE",
    "TRUTHY_REAL",
]


class BinaryForensicsDataset(Dataset):
    """Real/fake ``ImageFolder`` whose labels are pinned, not sort-dependent.

    ``real`` is class 0 and ``fake`` is class 1. With ``num_classes=1`` the
    model emits a single logit and fake should score high; with``num_classes=2``
    fake is index 1. Both cases agree with "fake is the positive class".
    """

    def __init__(self, root: str, transform=None, fake_dirs: Sequence[str] = ("fake",)):
        self.root = root
        self.transform = transform
        self.fake_dirs = tuple(fake_dirs)

        real_dir, fake_dir = self._resolve_dirs(root)
        self.samples: List[Tuple[str, int]] = []
        for d, label in ((real_dir, 0), (fake_dir, 1)):
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    self.samples.append((p, label))
        if not self.samples:
            raise RuntimeError(f"no images found under {root}")
        self.targets = [t for _, t in self.samples]
        self.classes = ["real", "fake"]
        self.class_to_idx = {"real": 0, "fake": 1}

    def _resolve_dirs(self, root: str) -> Tuple[str, str]:
        entries = [e for e in sorted(os.listdir(root))
                   if os.path.isdir(os.path.join(root, e))]

        # the pairing rule lives in src/layout.py so that scripts/prepare_data.py
        # stages exactly the pairs this loader will accept, no more and no less
        found = pair_class_dirs(entries)
        if found is not None:
            real_name, fake_name, exact = found
            if not exact:
                print(f"[warn] {root}: unrecognised class folders {entries}; "
                      f"treating {real_name!r} as real, {fake_name!r} as fake")
            return (os.path.join(root, real_name), os.path.join(root, fake_name))
        raise RuntimeError(
            f"{root}: expected exactly one real-like and one fake-like "
            f"subfolder, found {entries}. Rearrange to "
            f"<generator>/{{real,fake}}/"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, target = self.samples[i]
        # inlined torchvision.datasets.folder.default_loader: this project only
        # needs torch + Pillow, and keeping torchvision out means the package
        # imports on a torch-only CPU box
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def list_generators(root: str) -> List[str]:
    """Sorted generator subdirectories under a benchmark root."""
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    return sorted(e for e in os.listdir(root)
                  if os.path.isdir(os.path.join(root, e)))


def describe_split(root: str) -> List[Tuple[str, int, int]]:
    """(generator, n_real, n_fake) for every generator under ``root``."""
    out = []
    for gen in list_generators(root):
        g = os.path.join(root, gen)
        try:
            ds = BinaryForensicsDataset(g)
        except RuntimeError:
            continue
        n_real = sum(1 for _, t in ds.samples if t == 0)
        out.append((gen, n_real, len(ds) - n_real))
    return out
