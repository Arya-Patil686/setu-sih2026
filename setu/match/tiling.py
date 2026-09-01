"""Tiling for images far larger than a network's input size.

The networks of track A expect 640 to 840 px. Lunar orbital products are orders of
magnitude larger. The saving grace is that S1 has already co-registered the two grids to
within a few hundred pixels, so which reference tile pairs with which source tile is
known in advance - all-pairs matching is never performed and never needs to be.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TilePair:
    """One source tile and the reference tile it is known to overlap."""

    src_row: int
    src_col: int
    ref_row: int
    ref_col: int
    size: int

    @property
    def src_origin(self) -> tuple[float, float]:
        """(x, y) offset to lift tile-local coordinates back to the full image."""
        return float(self.src_col), float(self.src_row)

    @property
    def ref_origin(self) -> tuple[float, float]:
        return float(self.ref_col), float(self.ref_row)


def tile_origins(height: int, width: int, size: int, overlap: float = 0.5) -> list[tuple[int, int]]:
    """Top-left origins of a tiling with the given fractional overlap.

    The final row and column are pulled flush with the image edge rather than being
    left short, so the border gets the same coverage as the interior. Without that,
    uniformity scores are systematically penalised at the margins.
    """
    step = max(1, int(round(size * (1.0 - overlap))))
    rows = list(range(0, max(height - size, 0) + 1, step))
    cols = list(range(0, max(width - size, 0) + 1, step))
    if not rows or rows[-1] != max(height - size, 0):
        rows.append(max(height - size, 0))
    if not cols or cols[-1] != max(width - size, 0):
        cols.append(max(width - size, 0))
    return [(r, c) for r in sorted(set(rows)) for c in sorted(set(cols))]


def plan_tiles(
    src_shape: tuple[int, int],
    ref_shape: tuple[int, int],
    size: int = 640,
    overlap: float = 0.5,
    residual_px: float = 0.0,
    scale: float = 1.0,
) -> list[TilePair]:
    """Pair source tiles with their known reference counterparts.

    `residual_px` is S1's estimate of how far the pre-alignment can still be out. The
    reference tile is enlarged by that margin, so the true correspondence is inside the
    tile even in the worst case, without searching anywhere it cannot be.
    """
    sh, sw = src_shape[:2]
    rh, rw = ref_shape[:2]
    pad = int(np.ceil(residual_px))
    pairs: list[TilePair] = []

    for r, c in tile_origins(sh, sw, min(size, sh, sw), overlap):
        rr = int(round(r * scale)) - pad
        cc = int(round(c * scale)) - pad
        pairs.append(TilePair(
            src_row=r, src_col=c,
            ref_row=int(np.clip(rr, 0, max(rh - size, 0))),
            ref_col=int(np.clip(cc, 0, max(rw - size, 0))),
            size=size,
        ))
    return pairs


def crop(image: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    """Crop a tile, padding by reflection if it runs off the edge."""
    h, w = image.shape[:2]
    r1, c1 = min(row + size, h), min(col + size, w)
    tile = image[row:r1, col:c1]
    if tile.shape[0] == size and tile.shape[1] == size:
        return np.ascontiguousarray(tile)
    pad_r, pad_c = size - tile.shape[0], size - tile.shape[1]
    return np.pad(tile, ((0, pad_r), (0, pad_c)), mode="reflect" if min(tile.shape[:2]) > 1 else "constant")


def iter_tiles(
    src: np.ndarray,
    ref: np.ndarray,
    size: int = 640,
    overlap: float = 0.5,
    residual_px: float = 0.0,
    scale: float = 1.0,
) -> Iterator[tuple[np.ndarray, np.ndarray, TilePair]]:
    """Yield co-located tile pairs ready for a matcher."""
    for pair in plan_tiles(src.shape, ref.shape, size, overlap, residual_px, scale):
        yield (
            crop(src, pair.src_row, pair.src_col, pair.size),
            crop(ref, pair.ref_row, pair.ref_col, pair.size),
            pair,
        )


def _highpass_emphasis(size: int) -> np.ndarray:
    """Reddy-Chatterji high-pass emphasis filter for the magnitude spectrum.

    Log-polar sampling puts most of its bins near the origin, where the spectrum is
    dominated by the low frequencies that carry almost no orientation information. This
    filter suppresses that region so the correlation is driven by the oriented structure
    further out, and it is the difference between recovering rotation to a degree and
    not recovering it at all.
    """
    x = np.linspace(-0.5, 0.5, size)[None, :]
    y = np.linspace(-0.5, 0.5, size)[:, None]
    cos_term = np.cos(np.pi * x) * np.cos(np.pi * y)
    return ((1.0 - cos_term) * (2.0 - cos_term)).astype(np.float32)


def _log_polar_spectrum(image: np.ndarray, size: int) -> np.ndarray:
    """Log-magnitude spectrum of an image, resampled into log-polar coordinates."""
    import cv2

    img = cv2.resize(np.asarray(image, dtype=np.float32), (size, size))
    # Windowing before the FFT: the spectral leakage from a hard image edge is a strong
    # cross shape in the spectrum and would otherwise dominate the correlation peak.
    win = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(img * win)))
    mag = np.log1p(mag) * _highpass_emphasis(size)

    centre = (size / 2.0, size / 2.0)
    max_radius = size / 2.0
    flags = cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS | cv2.WARP_POLAR_LOG
    return cv2.warpPolar(mag, (size, size), centre, max_radius, flags)


def estimate_rotation_logpolar(src: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """Recover relative rotation and scale by phase correlation in log-polar space.

    Lunar orbital strips can differ in heading by up to 180 degrees. When the products'
    along-track azimuths are in the metadata that is the answer; this is the fallback
    for when they are not. A rotation and a scale become two translations under the
    log-polar transform, which phase correlation then reads off directly.

    Only recoverable modulo 180 degrees: the magnitude spectrum of a real image is
    centrosymmetric, so a heading and its reverse are indistinguishable here. The caller
    disambiguates with metadata, or by trying both hypotheses and keeping the one that
    yields more matches.
    """
    from skimage.registration import phase_cross_correlation

    size = int(min(src.shape[0], src.shape[1], ref.shape[0], ref.shape[1]))
    size = max(64, size - (size % 2))

    la = _log_polar_spectrum(src, size)
    lb = _log_polar_spectrum(ref, size)

    shift, _, _ = phase_cross_correlation(lb, la, upsample_factor=8, normalization=None)

    # warpPolar lays angle along rows over a full turn, and log-radius along columns
    # with kn = width / log(max_radius).
    d_angle = float(shift[0]) * 360.0 / size
    kn = size / np.log(max(size / 2.0, 2.0))
    d_scale = float(np.exp(shift[1] / kn))
    return d_angle % 180.0, d_scale
