"""S2c - cross-modality handling for the IIRS imaging spectrometer.

A 250-band cube is never handed to a matcher. Below about 2.0 um the IIRS signal is
dominated by reflected sunlight, so the scene looks morphologically like a panchromatic
image; above 2.5 um thermal emission takes over and the scene structure changes
entirely. The 900-1600 nm window used here is the reflected-solar range that published
IIRS photometric work also operates in.
"""

from __future__ import annotations

import numpy as np


def destripe_columns(cube: np.ndarray, axis: int = 1) -> np.ndarray:
    """Remove the column-wise offset that every pushbroom spectrometer carries.

    Each detector column has its own gain and offset, so a scene-median subtraction per
    column removes the stripe without touching real along-track structure.
    """
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim == 2:
        col_med = np.nanmedian(arr, axis=0, keepdims=True)
        return (arr - col_med + np.nanmedian(arr)).astype(np.float32)

    out = np.empty_like(arr)
    scene_med = np.nanmedian(arr, axis=(0, 1), keepdims=True)
    col_med = np.nanmedian(arr, axis=0, keepdims=True)
    out[:] = arr - col_med + scene_med
    return out.astype(np.float32)


def bad_pixel_mask(cube: np.ndarray, n_sigma: float = 6.0) -> np.ndarray:
    """Per-band mask of dead and hot detector elements."""
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    med = np.nanmedian(arr, axis=(0, 1), keepdims=True)
    mad = np.nanmedian(np.abs(arr - med), axis=(0, 1), keepdims=True)
    sigma = np.maximum(mad / 0.6745, 1e-6)
    return (np.abs(arr - med) > n_sigma * sigma) | ~np.isfinite(arr)


def pseudo_panchromatic(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray | None = None,
    lo_nm: float = 900.0,
    hi_nm: float = 1600.0,
    destripe: bool = True,
) -> np.ndarray:
    """pan_iirs = mean over the reflected-solar bands, after masking and destriping.

    Bad pixels are masked before averaging rather than after, because a single hot
    detector element averaged into the stack puts a bright column through the whole
    pseudo-pan image and the phase-congruency transform of S2b then treats it as a
    strong, perfectly straight edge.
    """
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim == 2:
        return arr.copy()

    n_bands = arr.shape[2]
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(800.0, 5000.0, n_bands)
    wl = np.asarray(wavelengths_nm, dtype=np.float64)

    sel = (wl >= lo_nm) & (wl <= hi_nm)
    if not sel.any():
        # No band in the reflected-solar window: fall back to the shortest quarter of
        # the range, which is still further from the thermal crossover than the mean.
        sel = np.zeros(n_bands, dtype=bool)
        sel[: max(1, n_bands // 4)] = True

    sub = arr[:, :, sel]
    if destripe:
        sub = destripe_columns(sub)

    bad = bad_pixel_mask(sub)
    masked = np.where(bad, np.nan, sub)
    with np.errstate(invalid="ignore"):
        pan = np.nanmean(masked, axis=2)

    if np.isnan(pan).any():
        pan = np.where(np.isnan(pan), float(np.nanmedian(pan)), pan)
    return pan.astype(np.float32)


def thermal_crossover_nm() -> float:
    """Where reflected solar gives way to thermal emission on the lunar dayside.

    Around 2.5 um at typical dayside temperatures. Bands beyond this are excluded from
    the pseudo-pan synthesis; correcting them properly needs the full thermal model
    (Verma et al. 2022) and is not required for registration.
    """
    return 2500.0
