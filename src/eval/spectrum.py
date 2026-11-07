"""Spectral descriptors used as (a) training-free baselines and (b) evidence.

Everything here is plain numpy so it can run and be tested without torch, and
so the reported statistics are reproducible outside the training stack.

The two descriptors implemented are the ones that carry the actual argument of
the project:

``radial_power_profile``
    Azimuthally averaged log power spectrum, i.e. the 1-D curve
    ``log |F(u, v)|^2`` collapsed over angle. GAN and diffusion decoders leave
    a characteristic comb in this curve at frequencies tied to their up-sampling
    factors, which is exactly what a "frequency fingerprint" claim rests on.

``hf_energy_ratio``
    Fraction of AC power above a normalised radial cutoff. A single number per
    image: the simplest possible "is it high-frequency-rich" test, and the
    scalar that a training-free baseline can threshold on.

Both operate on a float image in [0, 1] with shape (H, W, 3) or (H, W).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "to_luma",
    "log_power_spectrum",
    "radial_power_profile",
    "hf_energy_ratio",
    "azimuthal_average",
    "spectral_distance",
    "pairwise_spectral_distance",
    "descriptor_vector",
]


def to_luma(img: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma, float64 in [0, 1].

    Uses the same weights as PIL's ``convert("L")`` so numbers agree with what
    a normal image pipeline would report.
    """
    a = np.asarray(img)
    if a.ndim == 2:
        out = a.astype(np.float64)
    elif a.ndim == 3:
        if a.shape[2] == 1:
            out = a[..., 0].astype(np.float64)
        elif a.shape[2] >= 3:
            rgb = a[..., :3].astype(np.float64)
            out = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        else:
            raise ValueError(f"unsupported channel count: {a.shape[2]}")
    else:
        raise ValueError(f"unsupported shape: {a.shape}")

    if out.max() > 1.5:  # looks like uint8 / 0-255
        out = out / 255.0
    return out


def log_power_spectrum(img: np.ndarray, window: bool = True) -> np.ndarray:
    """``log1p(|rfft2(windowed luma)|^2)`` with the DC term removed.

    A Hann window is applied by default because the image border otherwise
    produces a strong cross in the spectrum that swamps the radial profile.
    """
    x = to_luma(img)
    h, w = x.shape
    x = x - x.mean()
    if window:
        wy = np.hanning(h)[:, None]
        wx = np.hanning(w)[None, :]
        x = x * wy * wx
    f = np.fft.rfft2(x, norm="ortho")
    p = np.abs(f) ** 2
    p[0, 0] = 0.0
    return np.log1p(p)


def _radius_grid(h: int, w: int) -> np.ndarray:
    """Normalised radial frequency grid, shaped like ``rfft2`` output.

    ``w`` must be the *image* width: ``np.fft.rfftfreq`` already halves it.
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)


def radial_power_profile(
    img: np.ndarray,
    n_bins: int = 64,
    r_max: float = 0.5,
    window: bool = True,
    exclude_dc: bool = True,
) -> np.ndarray:
    """Azimuthally averaged log power, sampled on ``n_bins`` radial bins.

    Returns an array of length ``n_bins`` where index ``i`` corresponds to
    normalised frequency ``r_max * (i + 0.5) / n_bins``.

    ``exclude_dc`` drops the exact zero-frequency sample before averaging. It
    matters: with a fine binning the innermost annulus can contain *only* the
    DC term, which `log_power_spectrum` deliberately zeroes, so that bin would
    read 0 and dominate every ratio statistic computed from the profile.
    """
    lp = log_power_spectrum(img, window=window)
    h, w = to_luma(img).shape
    r = _radius_grid(h, w)
    if exclude_dc:
        keep = r > 0
        lp, r = lp[keep], r[keep]
    return azimuthal_average(lp, r, n_bins=n_bins, r_max=r_max)


def azimuthal_average(
    values: np.ndarray,
    radius: np.ndarray,
    n_bins: int = 64,
    r_max: float = 0.5,
) -> np.ndarray:
    """Mean of ``values`` within each annulus, NaN-free (empty bins fall back to
    linear interpolation between populated neighbours)."""
    idx = np.floor(radius / r_max * n_bins).astype(np.int64)
    idx = np.clip(idx, 0, n_bins - 1)

    sums = np.bincount(idx.ravel(), weights=values.ravel(), minlength=n_bins)
    counts = np.bincount(idx.ravel(), minlength=n_bins).astype(np.float64)
    out = np.full(n_bins, np.nan)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    if not nz.all():
        if nz.sum() >= 2:
            out[~nz] = np.interp(np.flatnonzero(~nz),
                                 np.flatnonzero(nz), out[nz])
        else:
            out[~nz] = np.nanmean(out[nz]) if nz.any() else 0.0
    return out


def hf_energy_ratio(img: np.ndarray, cutoff: float = 0.15,
                    window: bool = True) -> float:
    """Fraction of AC log-power located above ``cutoff`` normalised frequency."""
    lp = log_power_spectrum(img, window=window)
    h, w = to_luma(img).shape
    r = _radius_grid(h, w)
    total = lp[r > 0].sum()
    if total <= 0:
        return float("nan")
    return float(lp[r > cutoff].sum() / total)


def spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised L1 distance between two radial profiles."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = np.abs(a).mean() + np.abs(b).mean() + 1e-12
    return float(np.abs(a - b).sum() / (denom * a.size))


def pairwise_spectral_distance(profiles: Sequence[np.ndarray]) -> np.ndarray:
    """Full distance matrix (does not divide by 2, diagonal is exactly 0)."""
    n = len(profiles)
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = spectral_distance(profiles[i], profiles[j])
            m[i, j] = m[j, i] = d
    return m


def descriptor_vector(img: np.ndarray, n_bins: int = 64,
                      cutoff: float = 0.15) -> Dict[str, float]:
    """Scalars + profile summary for one image, ready to go into a CSV row."""
    prof = radial_power_profile(img, n_bins=n_bins)
    return {
        "hf_ratio": hf_energy_ratio(img, cutoff=cutoff),
        "profile_mean": float(np.nanmean(prof)),
        "profile_std": float(np.nanstd(prof)),
        "profile_peak_bin": float(np.nanargmax(prof)),
    }
