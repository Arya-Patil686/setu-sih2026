"""S7 - resampling the source into the reference geometry.

Lanczos for the delivered product, bicubic for previews. The distinction is not
cosmetic: Lanczos preserves high-frequency detail that a downstream user may want to
measure, and costs enough that it is worth avoiding on the dozen preview images a report
generates.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

RESAMPLING = {
    "lanczos": cv2.INTER_LANCZOS4,
    "cubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
}


def warp_global(
    source: np.ndarray,
    H: np.ndarray,
    out_shape: tuple[int, int],
    resample: str = "lanczos",
    nodata: float = 0.0,
) -> np.ndarray:
    """Apply the global transform, mapping source pixels into the reference grid."""
    flags = RESAMPLING.get(resample, cv2.INTER_LANCZOS4)
    return cv2.warpPerspective(
        np.ascontiguousarray(source, dtype=np.float32),
        np.asarray(H, dtype=np.float64),
        (out_shape[1], out_shape[0]),
        flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=nodata,
    )


def warp_with_local(
    source: np.ndarray,
    H: np.ndarray,
    out_shape: tuple[int, int],
    local_model: Any | None = None,
    jitter: Any | None = None,
    resample: str = "lanczos",
    nodata: float = 0.0,
    grid_step: int = 16,
) -> np.ndarray:
    """Apply the global transform plus the local correction and the jitter spline.

    The local correction is evaluated on a coarse grid and interpolated up rather than at
    every output pixel. A thin-plate spline costs O(n_control) per evaluation, so a
    full-resolution evaluation of a 400-control-point spline over a 4000 x 4000 grid is
    minutes of work for a correction that is smooth by construction and is therefore
    identical either way to well below a hundredth of a pixel.
    """
    if local_model is None and jitter is None:
        return warp_global(source, H, out_shape, resample, nodata)

    h, w = out_shape[:2]
    H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))

    ys = np.arange(0, h + grid_step, grid_step, dtype=np.float64)
    xs = np.arange(0, w + grid_step, grid_step, dtype=np.float64)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    ref_pts = np.column_stack([gx.ravel(), gy.ravel()])

    from setu.bench.generate import apply_h

    src_pts = apply_h(H_inv, ref_pts)

    if local_model is not None and getattr(local_model, "improves", False):
        # The local model predicts the correction in the *source* frame, so it is applied
        # there before the point is used to sample.
        src_pts = src_pts - local_model.predict(src_pts)
    if jitter is not None:
        src_pts = src_pts - jitter.predict(src_pts[:, 1])

    map_x = src_pts[:, 0].reshape(gy.shape).astype(np.float32)
    map_y = src_pts[:, 1].reshape(gy.shape).astype(np.float32)
    map_x = cv2.resize(map_x, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y, (w, h), interpolation=cv2.INTER_LINEAR)

    return cv2.remap(
        np.ascontiguousarray(source, dtype=np.float32), map_x, map_y,
        interpolation=RESAMPLING.get(resample, cv2.INTER_LANCZOS4),
        borderMode=cv2.BORDER_CONSTANT, borderValue=nodata,
    )


def checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 64) -> np.ndarray:
    """Checkerboard composite of two co-registered images.

    Misregistration shows up as broken edges at the tile boundaries, which a viewer
    reads instantly and which no table conveys.
    """
    a = _norm(a)
    b = _norm(b)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((yy // tile) + (xx // tile)) % 2).astype(bool)
    return np.where(mask, a, b).astype(np.float32)


def swipe(a: np.ndarray, b: np.ndarray, position: float = 0.5) -> np.ndarray:
    """Split-screen composite with the seam at a fractional horizontal position."""
    a = _norm(a)
    b = _norm(b)
    split = int(np.clip(position, 0.0, 1.0) * a.shape[1])
    out = b.copy()
    out[:, :split] = a[:, :split]
    return out.astype(np.float32)


def blink_frames(a: np.ndarray, b: np.ndarray, n: int = 8) -> list[np.ndarray]:
    """Cross-fade frames between two images, for the animated blink comparison."""
    a = _norm(a)
    b = _norm(b)
    return [((1 - t) * a + t * b).astype(np.float32)
            for t in np.concatenate([np.linspace(0, 1, n // 2), np.linspace(1, 0, n - n // 2)])]


def _norm(image: np.ndarray) -> np.ndarray:
    """Percentile stretch to [0, 1], so two images with different radiometry compare fairly."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    finite = img[np.isfinite(img) & (img != 0)]
    if finite.size == 0:
        return np.zeros_like(img)
    lo, hi = np.percentile(finite, [1, 99])
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def to_png_bytes(image: np.ndarray, max_px: int = 1400) -> bytes:
    """8-bit PNG bytes of a float image, downsampled to fit `max_px`."""
    img = _norm(image)
    h, w = img.shape[:2]
    scale = min(1.0, max_px / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", (img * 255).astype(np.uint8))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
