"""S4 step 1 - upsampled phase correlation.

The matrix-multiply DFT upsampling of Guizar-Sicairos et al., as implemented in
`skimage.registration.phase_cross_correlation`. It is the standard, well-tested route to
roughly 0.02 to 0.1 px on clean data, and it costs almost nothing because the upsampling
is evaluated only in a small neighbourhood of the peak rather than over a zero-padded FFT.

Refinement runs on the *structural* representations rather than the raw images. That is
the whole point of S2b: a structural map is illumination-stable, so the correlation peak
sits at the geometric correspondence rather than at whatever the shadows happen to align.
"""

from __future__ import annotations

import numpy as np
from skimage.registration import phase_cross_correlation


def subpixel_patch(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray | None:
    """Square patch centred *exactly* on (x, y), including its fractional part.

    This matters more than it looks. `extract_patch` centres on the rounded coordinate,
    so a shift measured against it is a shift relative to `round(p)`, not to `p` - and
    discarding that rounding, twice, once for phase correlation and again for
    least-squares matching, injects up to a pixel of systematic error into every
    refinement. Since the entire point of S4 is to be correct in the third decimal place,
    the patch has to be sampled where it says it is.
    """
    import cv2

    h, w = image.shape[:2]
    half = size / 2.0
    if x - half < 1 or y - half < 1 or x + half >= w - 1 or y + half >= h - 1:
        return None

    # Output pixel (u, v) samples the source at (x, y) + (u - half, v - half), so the
    # patch centre lands on (x, y) with no rounding anywhere.
    M = np.array([[1.0, 0.0, x - half], [0.0, 1.0, y - half]], dtype=np.float64)
    patch = cv2.warpAffine(
        np.ascontiguousarray(image, dtype=np.float32), M, (size, size),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT101,
    )
    return patch if np.isfinite(patch).all() else None


def extract_patch(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray | None:
    """Square patch centred on (x, y), or None if it would cross the image edge."""
    half = size // 2
    c0, r0 = int(round(x)) - half, int(round(y)) - half
    if c0 < 0 or r0 < 0 or r0 + size > image.shape[0] or c0 + size > image.shape[1]:
        return None
    patch = np.asarray(image[r0:r0 + size, c0:c0 + size], dtype=np.float32)
    return patch if np.isfinite(patch).all() else None


def _window(size: int) -> np.ndarray:
    """Hann window, applied to both patches before correlation.

    Without it, the discontinuity at the patch border leaks into the spectrum as a
    strong cross and biases the peak towards zero shift - which looks like a
    well-converged refinement and is not one.
    """
    w = np.hanning(size)
    return np.outer(w, w).astype(np.float32)


def refine_phase(
    src_repr: np.ndarray,
    ref_repr: np.ndarray,
    pt_src: np.ndarray,
    pt_ref: np.ndarray,
    patch: int = 64,
    upsample_factor: int = 50,
) -> tuple[np.ndarray, dict[str, float]] | None:
    """Refine one correspondence by upsampled phase correlation.

    Returns the corrected reference position and the diagnostics needed downstream:
    the shift applied, the normalised correlation peak, and the phase-correlation error
    that skimage reports.
    """
    a = extract_patch(src_repr, pt_src[0], pt_src[1], patch)
    b = extract_patch(ref_repr, pt_ref[0], pt_ref[1], patch)
    if a is None or b is None:
        return None
    if a.std() < 1e-8 or b.std() < 1e-8:
        return None

    win = _window(patch)
    a = (a - a.mean()) * win
    b = (b - b.mean()) * win

    try:
        shift, error, phasediff = phase_cross_correlation(
            b, a, upsample_factor=upsample_factor, normalization="phase"
        )
    except Exception:
        return None

    dy, dx = float(shift[0]), float(shift[1])
    if not np.isfinite(dy) or not np.isfinite(dx):
        return None
    # A shift larger than a quarter of the patch means the correlation locked onto a
    # different feature, not a refinement of this one.
    if max(abs(dy), abs(dx)) > patch / 4.0:
        return None

    return (
        np.array([pt_ref[0] + dx, pt_ref[1] + dy], dtype=np.float64),
        {"dx": dx, "dy": dy, "pc_error": float(error), "phasediff": float(phasediff)},
    )


def correlation_surface(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation surface of two equal-sized patches, for diagnostics."""
    fa = np.fft.fft2(a - a.mean())
    fb = np.fft.fft2(b - b.mean())
    cross = fa * np.conj(fb)
    denom = np.abs(cross)
    cross = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 1e-12)
    return np.fft.fftshift(np.real(np.fft.ifft2(cross)))
