"""Residual-operator family for AIGC forensics.

Why this module exists
----------------------
The official NPR implementation (chuangchuangtan/NPR-DeepfakeDetection,
networks/resnet.py) feeds the network the following tensor instead of the image:

    NPR = x - upsample_nearest(downsample_nearest(x))
    x_in = conv1(NPR * 2.0 / 3.0)

That is, before any learning happens, the image is passed through a
fixed high-pass residual operator built from a nearest-neighbour 2x
down/up round trip. The 2/3 factor is an empirical normalisation that we
keep verbatim so the released `NPR.pth` checkpoint stays bit-compatible.

This module turns that single hard-coded operator into a *family*, so the
central question of the project becomes measurable:

    Does NPR generalise because it extracts high frequencies, or because its
    nearest-neighbour round trip happens to mirror the up-sampling operator
    that GAN/diffusion decoders use to produce an image?

Prediction of the "operator-matching" hypothesis: nearest-neighbour residual
should win; every other high-pass operator (bilinear/bicubic round trip,
Laplacian, block-DCT low-pass removal, FFT radial low-pass removal) should
lose, even though all of them are "also high-pass".

Every operator returns a residual with the same shape as the input
(B, C, H, W). Odd spatial sizes are cropped by one pixel, mirroring the fix
the official README documents for AIGCDetectBenchmark.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "make_residual_op",
    "REGISTRY",
    "OPERATOR_DESCRIPTIONS",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _even(x: torch.Tensor) -> torch.Tensor:
    """Crop the last row/column when side length is odd (official README note)."""
    h, w = x.shape[-2:]
    h2, w2 = h - (h % 2), w - (w % 2)
    if (h2, w2) != (h, w):
        x = x[..., :h2, :w2]
    return x


def _round_trip(x: torch.Tensor, mode: str) -> torch.Tensor:
    """downsample by 2 then upsample by 2, using `mode` for both directions."""
    x = _even(x)
    kwargs = {"recompute_scale_factor": None}
    if mode != "nearest":
        kwargs["align_corners"] = False
    down = F.interpolate(x, scale_factor=0.5, mode=mode, **kwargs)
    up = F.interpolate(down, scale_factor=2.0, mode=mode, **kwargs)
    return up


def _gaussian_kernel(device, dtype, sigma: float = 1.0) -> torch.Tensor:
    r = int(math.ceil(3 * sigma))
    coords = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _gaussian_blur(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Separable Gaussian blur, applied independently per channel.

    Edges are padded by *replication*, not by zeros: zero padding would make a
    flat image "fade out" near the border, so the Laplacian residual of a
    constant image would be non-zero and the operator would inject a fake
    frame-shaped edge -- exactly the kind of artifact a forensic high-pass
    must not manufacture.
    """
    k = _gaussian_kernel(x.device, x.dtype, sigma)
    r = (k.numel() - 1) // 2
    b, c, h, w = x.shape
    y = x.reshape(b * c, 1, h, w)
    y = F.pad(y, (r, r, 0, 0), mode="replicate")
    y = F.conv2d(y, k.view(1, 1, 1, -1))
    y = F.pad(y, (0, 0, r, r), mode="replicate")
    y = F.conv2d(y, k.view(1, 1, -1, 1))
    return y.view(b, c, h, w)


def _dct_basis(n: int, device, dtype) -> torch.Tensor:
    """Orthonormal DCT-II basis matrix, rows indexed by frequency u."""
    k = torch.arange(n, device=device, dtype=dtype)
    i = torch.arange(n, device=device, dtype=dtype)
    c = torch.cos(math.pi * (2 * i[None, :] + 1) * k[:, None] / (2.0 * n))
    c = c * math.sqrt(2.0 / n)
    c[0] = c[0] / math.sqrt(2.0)
    return c


def _radial_mask(h: int, w: int, cutoff: float, device, dtype) -> torch.Tensor:
    """Low-pass mask in normalised radial frequency, shape (1, 1, h, w//2+1)."""
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype).view(h, 1)
    fx = torch.fft.rfftfreq(w, device=device, dtype=dtype).view(1, w // 2 + 1)
    radius = torch.sqrt(fy ** 2 + fx ** 2)
    return (radius <= cutoff).to(dtype).view(1, 1, h, w // 2 + 1)


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #
class ResidualOp(nn.Module):
    """Base class. `scale` multiplies the residual before it reaches the net."""

    name: str = "base"
    scale: float = 1.0

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.residual(x)
        return r if self.scale == 1.0 else r * self.scale


class IdentityOp(ResidualOp):
    """No high-pass at all: the network sees the raw image.

    This is the ablation that tells you how much the residual operator itself
    contributes, independent of any frequency content.
    """

    name = "identity"

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return _even(x)


class NearestRoundTripOp(ResidualOp):
    """Official NPR operator. `x - up_nearest(down_nearest(x))`, scale 2/3."""

    name = "npr_nearest"
    scale = 2.0 / 3.0

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        x = _even(x)
        return x - _round_trip(x, "nearest")


class BilinearRoundTripOp(ResidualOp):
    """Same round trip, bilinear interpolation. Mismatched with most decoders."""

    name = "npr_bilinear"
    scale = 2.0 / 3.0

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        x = _even(x)
        return x - _round_trip(x, "bilinear")


class BicubicRoundTripOp(ResidualOp):
    name = "npr_bicubic"
    scale = 2.0 / 3.0

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        x = _even(x)
        return x - _round_trip(x, "bicubic")


class BoxRoundTripOp(ResidualOp):
    """2x2 average pooling then nearest upsample.

    Separates two factors that nearest round trip confounds: *point sampling*
    (keeps one pixel of each 2x2 block) versus *low-pass then upsample*.
    """

    name = "npr_box"
    scale = 2.0 / 3.0

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        x = _even(x)
        down = F.avg_pool2d(x, kernel_size=2, stride=2)
        up = F.interpolate(down, scale_factor=2.0, mode="nearest")
        return x - up


class LaplacianOp(ResidualOp):
    """Classical high-pass: x - GaussianBlur(x).

    The textbook 'high-frequency image'. If NPR's advantage were merely
    'being high-pass', this operator should match it.
    """

    name = "laplacian"

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        x = _even(x)
        return x - _gaussian_blur(x, sigma=1.0)


class DCTLowpassOp(ResidualOp):
    """Remove the block-DCT low frequencies, keep the residual.

    Residual = x - IDCT(lowpass_mask * DCT(x)) over 8x8 blocks, i.e. exactly
    the 'same image minus its JPEG-quality low frequencies' operator.
    `keep_ratio` is the fraction of radial DCT bandwidth passed through.
    """

    name = "dct_lowpass"
    block: int = 8
    keep_ratio: float = 0.35
    scale = 1.0

    def __init__(self, block: int = 8, keep_ratio: float = 0.35):
        super().__init__()
        self.block = block
        self.keep_ratio = keep_ratio
        basis = _dct_basis(block, torch.device("cpu"), torch.float32)
        u = torch.arange(block).view(-1, 1).expand(block, block).float()
        v = torch.arange(block).view(1, -1).expand(block, block).float()
        radius = torch.sqrt(u ** 2 + v ** 2)
        cut = self.keep_ratio * radius.max()
        mask = (radius <= cut).float()
        # non-persistent: deterministic buffers must not pollute state_dict,
        # otherwise an official NPR checkpoint reports spurious missing keys
        self.register_buffer("basis", basis, persistent=False)   # (p, p), row u
        self.register_buffer("mask", mask, persistent=False)     # (p, p), 1 = keep

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        p = self.block
        # replicate-pad up to a whole number of blocks, so the operator is
        # defined everywhere (224 is already divisible; odd test sizes are not)
        h, w = x.shape[-2:]
        ph, pw = (-h) % p, (-w) % p
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        b, c, hh, ww = xp.shape

        # unfold -> (B, C*p*p, L); view as (B, C, i, j, L) with i=row, j=col
        patches = F.unfold(xp, kernel_size=p, stride=p)
        n = patches.shape[-1]
        blocks = patches.view(b, c, p, p, n)

        # forward 2-D DCT-II per block: X = C @ M @ C^T
        t = torch.einsum("ui,bcijl->bcujl", self.basis, blocks)
        coef = torch.einsum("vj,bcujl->bcuvl", self.basis, t)

        # radial low-pass in the (u, v) coefficient plane
        coef = coef * self.mask.view(1, 1, p, p, 1)

        # inverse: M = C^T @ X @ C  (note: C is NOT symmetric, so the first
        # einsum must contract over u with basis index order (u, i), not (i, u))
        t = torch.einsum("ui,bcuvl->bcivl", self.basis, coef)
        rec = torch.einsum("vj,bcivl->bcijl", self.basis, t)

        rec = F.fold(rec.reshape(b, c * p * p, n), output_size=(hh, ww),
                     kernel_size=p, stride=p).view(b, c, hh, ww)
        return x - rec[..., :h, :w]


class FFTRadialOp(ResidualOp):
    """Remove the radially low-frequency content of the 2D DFT."""

    name = "fft_radial"
    cutoff: float = 0.15
    scale = 1.0

    def __init__(self, cutoff: float = 0.15):
        super().__init__()
        self.cutoff = cutoff

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        mask = _radial_mask(h, w, self.cutoff, x.device, f.real.dtype)
        low = f * mask
        x_low = torch.fft.irfft2(low, s=(h, w), dim=(-2, -1), norm="ortho")
        return x - x_low


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
REGISTRY: Dict[str, type] = {
    "identity": IdentityOp,
    "npr_nearest": NearestRoundTripOp,
    "npr_bilinear": BilinearRoundTripOp,
    "npr_bicubic": BicubicRoundTripOp,
    "npr_box": BoxRoundTripOp,
    "laplacian": LaplacianOp,
    "dct_lowpass": DCTLowpassOp,
    "fft_radial": FFTRadialOp,
}

OPERATOR_DESCRIPTIONS: Dict[str, str] = {
    "identity": "raw image, no high-pass (control)",
    "npr_nearest": "official NPR: x - up_near(down_near(x)), scale 2/3",
    "npr_bilinear": "x - up_bilin(down_bilin(x)), scale 2/3",
    "npr_bicubic": "x - up_bicub(down_bicub(x)), scale 2/3",
    "npr_box": "x - up_near(avgpool2(x)), scale 2/3",
    "laplacian": "x - gaussian_blur(x), sigma=1",
    "dct_lowpass": "x - IDCT(radial low-pass of 8x8 block DCT)",
    "fft_radial": "x - IFFT(radial low-pass of 2D DFT)",
}


def make_residual_op(name: str, **kwargs) -> ResidualOp:
    if name not in REGISTRY:
        raise KeyError(
            f"unknown residual op {name!r}; available: {sorted(REGISTRY)}"
        )
    cls = REGISTRY[name]
    if name in ("dct_lowpass", "fft_radial"):
        return cls(**kwargs)
    return cls()
