"""S2b - illumination-invariant structural representations.

Re-illumination removes the bulk of the appearance difference, but not all of it:
albedo, sensor MTF and DEM error survive. These transforms convert both images into
representations that depend on structure rather than on brightness, so the residual
difference stops mattering.

All four are implemented and all four are benchmarked, because which one wins depends
on the pairing - phase congruency is the strongest on clean data and the weakest on
noisy IIRS bands, which is exactly the trade-off the configuration flag exists for.

Kovesi's phase congruency is vendored here rather than taken from `phasepack`, which is
unmaintained and breaks on modern NumPy. The implementation follows the monogenic
formulation of the 1999 paper directly.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

EPS = 1e-10
StructuralKind = Literal["pc", "mim", "cfog", "lnift", "none"]


# ------------------------------------------------------------ filter plumbing

def _filter_grid(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised frequency grid with DC at the corner, matching `np.fft.fft2` layout."""
    u1 = np.fft.ifftshift((np.arange(cols) - cols // 2) / cols)[None, :]
    u2 = np.fft.ifftshift((np.arange(rows) - rows // 2) / rows)[:, None]
    radius = np.sqrt(u1**2 + u2**2)
    return radius, np.broadcast_to(u1, (rows, cols)), np.broadcast_to(u2, (rows, cols))


def _lowpass_butterworth(radius: np.ndarray, cutoff: float = 0.45, order: int = 15) -> np.ndarray:
    """Butterworth low-pass, applied to every log-Gabor to suppress corner artefacts."""
    return 1.0 / (1.0 + (radius / cutoff) ** (2 * order))


def _log_gabor_radial(radius: np.ndarray, wavelength: float, sigma_onf: float) -> np.ndarray:
    """Radial component of a log-Gabor filter, zeroed at DC."""
    fo = 1.0 / wavelength
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.exp(-((np.log(radius / fo)) ** 2) / (2 * np.log(sigma_onf) ** 2))
    lg[0, 0] = 0.0
    return np.nan_to_num(lg)


def _angular_spread(u1: np.ndarray, u2: np.ndarray, angle: float, n_orient: int) -> np.ndarray:
    """Angular component of a log-Gabor, as a Gaussian in the angular distance.

    Angular distance is computed through `arctan2` of the cross and dot products rather
    than a raw angle difference, which keeps it well-behaved across the wrap at pi.
    """
    sin_t, cos_t = np.sin(angle), np.cos(angle)
    ds = u2 * cos_t - u1 * sin_t
    dc = u1 * cos_t + u2 * sin_t
    dtheta = np.abs(np.arctan2(ds, dc))
    dtheta = np.minimum(dtheta * n_orient / 2.0, np.pi)
    return (np.cos(dtheta) + 1.0) / 2.0


def _prep(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return np.nan_to_num(img, nan=float(np.nanmedian(img)) if np.isfinite(img).any() else 0.0)


# ---------------------------------------------------------- phase congruency

def phase_congruency(
    image: np.ndarray,
    n_scale: int = 4,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_onf: float = 0.55,
    k: float = 2.0,
    cutoff: float = 0.5,
    g: float = 10.0,
    noise_adaptive: bool = True,
) -> dict[str, np.ndarray]:
    """Monogenic phase congruency (Kovesi 1999).

    PC responds to points of maximal phase alignment across scales. Because phase is
    independent of both contrast and mean brightness, a crater rim registers the same
    whether it is lit from the east at 15 degrees or from the west at 60 - which is
    precisely the invariance a gradient operator does not have.

    `k` is the noise-rejection parameter and it is not cosmetic. PC is *not* robust to
    noise: on low-SNR IIRS bands and inside OHRC shadows it manufactures edges out of
    nothing. With `noise_adaptive` the threshold is estimated from the smallest-scale
    filter response through the Rayleigh statistics of the noise, which is the
    difference between a usable map and a field of spurious detections.
    """
    img = _prep(image)
    rows, cols = img.shape
    IM = np.fft.fft2(img)

    radius, u1, u2 = _filter_grid(rows, cols)
    radius = radius.copy()
    radius[0, 0] = 1.0
    lp = _lowpass_butterworth(radius)

    # Riesz transform kernel: the monogenic signal's odd component.
    H = (1j * u1 - u2) / radius
    H[0, 0] = 0

    sum_an = np.zeros((rows, cols), np.float64)
    sum_f = np.zeros((rows, cols), np.float64)
    sum_h1 = np.zeros((rows, cols), np.float64)
    sum_h2 = np.zeros((rows, cols), np.float64)
    max_an = np.zeros((rows, cols), np.float64)
    tau = 0.0

    for s in range(n_scale):
        lg = _log_gabor_radial(radius, min_wavelength * mult**s, sigma_onf) * lp
        IMF = IM * lg
        f = np.real(np.fft.ifft2(IMF))
        h = np.fft.ifft2(IMF * H)
        h1, h2 = np.real(h), np.imag(h)

        an = np.sqrt(f * f + h1 * h1 + h2 * h2)
        sum_an += an
        sum_f += f
        sum_h1 += h1
        sum_h2 += h2
        max_an = np.maximum(max_an, an)

        if s == 0 and noise_adaptive:
            # The smallest scale is dominated by noise, and the amplitude of Gaussian
            # noise through a quadrature pair is Rayleigh-distributed, so its median
            # gives the Rayleigh parameter directly.
            tau = float(np.median(an)) / np.sqrt(np.log(4.0))

    energy = np.sqrt(sum_f**2 + sum_h1**2 + sum_h2**2)

    if noise_adaptive and tau > 0:
        total_tau = tau * (1 - (1 / mult) ** n_scale) / (1 - (1 / mult))
        noise_mean = total_tau * np.sqrt(np.pi / 2.0)
        noise_sigma = total_tau * np.sqrt((4 - np.pi) / 2.0)
        T = noise_mean + k * noise_sigma
    else:
        T = 0.0

    # Frequency-spread weighting: a response driven by a single scale is not congruency,
    # it is one filter firing, so it is down-weighted by a sigmoid on the spread.
    width = (sum_an / (max_an + EPS) - 1.0) / max(n_scale - 1, 1)
    weight = 1.0 / (1.0 + np.exp((cutoff - width) * g))

    pc = weight * np.maximum(energy - T, 0.0) / (sum_an + EPS)

    return {
        "pc": pc.astype(np.float32),
        "orientation": np.arctan2(-sum_h2, sum_h1).astype(np.float32),
        "phase": np.arctan2(sum_f, np.sqrt(sum_h1**2 + sum_h2**2)).astype(np.float32),
        "energy": energy.astype(np.float32),
        "noise_threshold": float(T),
    }


# --------------------------------------------------------- maximum index map

def log_gabor_bank(
    image: np.ndarray,
    n_orient: int = 6,
    n_scale: int = 4,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_onf: float = 0.55,
) -> np.ndarray:
    """Summed log-Gabor amplitude per orientation, shape (n_orient, h, w)."""
    img = _prep(image)
    rows, cols = img.shape
    IM = np.fft.fft2(img)
    radius, u1, u2 = _filter_grid(rows, cols)
    radius = radius.copy()
    radius[0, 0] = 1.0
    lp = _lowpass_butterworth(radius)

    radial = [_log_gabor_radial(radius, min_wavelength * mult**s, sigma_onf) * lp for s in range(n_scale)]
    out = np.zeros((n_orient, rows, cols), np.float32)

    for o in range(n_orient):
        spread = _angular_spread(u1, u2, o * np.pi / n_orient, n_orient)
        acc = np.zeros((rows, cols), np.float64)
        for rad in radial:
            resp = np.fft.ifft2(IM * (rad * spread))
            acc += np.abs(resp)
        out[o] = acc.astype(np.float32)
    return out


def maximum_index_map(
    image: np.ndarray,
    n_orient: int = 6,
    n_scale: int = 4,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    normalise: bool = True,
) -> np.ndarray:
    """MIM - the index of the maximum-responding orientation per pixel (RIFT family).

    The map is an *index*, not an amplitude, so a non-linear radiometric distortion that
    scales or even inverts contrast leaves it unchanged as long as the dominant local
    orientation survives. Gradient maps do not have that property, which is why RIFT
    beats gradient-based descriptors on multi-modal pairs.
    """
    bank = log_gabor_bank(image, n_orient, n_scale, min_wavelength, mult)
    mim = np.argmax(bank, axis=0).astype(np.float32)
    return (mim / max(n_orient - 1, 1)) if normalise else mim


def mim_response(image: np.ndarray, n_orient: int = 6, n_scale: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """MIM index map together with its peak amplitude, used to mask flat regions."""
    bank = log_gabor_bank(image, n_orient, n_scale)
    return np.argmax(bank, axis=0).astype(np.int32), np.max(bank, axis=0).astype(np.float32)


# ---------------------------------------------------------------------- CFOG

def cfog(
    image: np.ndarray,
    n_orient: int = 8,
    sigma: float = 2.0,
    normalise: bool = True,
) -> np.ndarray:
    """Channel features of oriented gradients, shape (h, w, n_orient).

    A dense pixel-wise descriptor tensor: gradients are projected onto orientation
    channels, each channel is smoothed spatially, and the channel axis is then smoothed
    too, which is what makes the descriptor tolerate small orientation errors. Because
    it is dense and separable, template matching against it can run as an FFT
    convolution, which is why S3 track B uses it as its similarity metric.
    """
    img = _prep(image)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    channels = np.empty((img.shape[0], img.shape[1], n_orient), np.float32)
    for o in range(n_orient):
        theta = o * np.pi / n_orient
        # Absolute projection: a gradient and its negation describe the same edge, and
        # on cross-modal pairs the sign flips routinely.
        proj = np.abs(gx * np.cos(theta) + gy * np.sin(theta))
        channels[:, :, o] = cv2.GaussianBlur(proj, (0, 0), sigma)

    # Smoothing along the orientation axis, wrapped, with the [1 2 1] kernel of the
    # original formulation.
    smoothed = (
        0.25 * np.roll(channels, 1, axis=2) + 0.5 * channels + 0.25 * np.roll(channels, -1, axis=2)
    ).astype(np.float32)

    if normalise:
        norm = np.sqrt((smoothed**2).sum(axis=2, keepdims=True)) + EPS
        smoothed = smoothed / norm
    return smoothed


def cfog_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel CFOG similarity between two descriptor tensors of the same shape."""
    return np.sum(a * b, axis=2)


# --------------------------------------------------------------------- LNIFT

def lnift(image: np.ndarray, window: int = 31) -> np.ndarray:
    """Local mean and standard-deviation normalisation.

    The cheap baseline. It removes the low-frequency illumination gradient and nothing
    else, which is enough to matter and is fast, so it is worth measuring against the
    more expensive transforms rather than assuming they win.
    """
    img = _prep(image)
    mean = uniform_filter(img, window)
    sq = uniform_filter(img * img, window)
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0)) + EPS
    return ((img - mean) / std).astype(np.float32)


# ------------------------------------------------------------------ dispatch

def structural_transform(
    image: np.ndarray,
    kind: StructuralKind = "mim",
    **kwargs,
) -> np.ndarray:
    """Apply one structural transform, returning a single-channel float32 map.

    CFOG is a tensor rather than a map, so for the single-channel contract it is
    collapsed to its dominant-channel index - the same quantity MIM produces, computed
    from gradients instead of log-Gabor responses. Track B uses the full tensor directly.
    """
    if kind == "none":
        return _prep(image)
    if kind == "pc":
        return phase_congruency(image, **kwargs)["pc"]
    if kind == "mim":
        return maximum_index_map(image, **kwargs)
    if kind == "lnift":
        return lnift(image, **kwargs)
    if kind == "cfog":
        tensor = cfog(image, **kwargs)
        return (np.argmax(tensor, axis=2) / max(tensor.shape[2] - 1, 1)).astype(np.float32)
    raise ValueError(f"unknown structural transform {kind!r}")


def normalise01(x: np.ndarray) -> np.ndarray:
    """Scale a map to [0, 1] robustly, for display and for correlation."""
    a = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(a[np.isfinite(a)], [1, 99]) if np.isfinite(a).any() else (0.0, 1.0)
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)
