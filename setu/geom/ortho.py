"""S1 step 3 - ortho-projection onto a common map grid.

For each output map pixel, its ground position is intersected with the shape model and
back-projected into the source image through the sensor model. Sampling the source at
those coordinates puts both images on the identical grid, which is what collapses scale
ratios of up to 160x and viewpoint differences of tens of degrees into a residual planar
misalignment.

Resampling direction is a rule, not a preference: the finer image is always downsampled
with area-averaging and the coarser one is never upsampled to meet it. Bicubic
upsampling of low-sun lunar imagery measurably *reduces* structural similarity, which
MoonMetaSync documents, and manufacturing detail that the sensor never recorded is the
one thing guaranteed to make a sub-pixel claim meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from pyproj import CRS
from rasterio.transform import Affine

from setu.geom.crs import lonlat_to_map, map_to_lonlat
from setu.geom.sensor_model import SensorModel, parallax_shift
from setu.types import Product


@dataclass
class OrthoGrid:
    """The common map grid both products are resampled onto."""

    crs: CRS
    transform: Affine
    height: int
    width: int
    gsd_m: float

    def map_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Map x and y of every output pixel centre."""
        cols, rows = np.meshgrid(np.arange(self.width), np.arange(self.height))
        x = self.transform.c + (cols + 0.5) * self.transform.a + (rows + 0.5) * self.transform.b
        y = self.transform.f + (cols + 0.5) * self.transform.d + (rows + 0.5) * self.transform.e
        return x.astype(np.float64), y.astype(np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "crs": self.crs.to_proj4(),
            "transform": list(self.transform)[:6],
            "height": self.height,
            "width": self.width,
            "gsd_m": round(self.gsd_m, 6),
        }


def build_grid(
    bounds_map: tuple[float, float, float, float],
    crs: CRS,
    gsd_m: float,
    max_size_px: int = 4096,
) -> OrthoGrid:
    """A north-up map grid covering the given bounds at the working GSD.

    If the requested extent would exceed `max_size_px`, the GSD is coarsened rather than
    the extent being cropped: a smaller area at full resolution would silently change
    what is being registered.
    """
    x0, y0, x1, y1 = bounds_map
    width = int(np.ceil((x1 - x0) / gsd_m))
    height = int(np.ceil((y1 - y0) / gsd_m))

    if max(width, height) > max_size_px:
        factor = max(width, height) / max_size_px
        gsd_m *= factor
        width = int(np.ceil((x1 - x0) / gsd_m))
        height = int(np.ceil((y1 - y0) / gsd_m))

    transform = Affine(gsd_m, 0.0, x0, 0.0, -gsd_m, y1)
    return OrthoGrid(crs=crs, transform=transform, height=max(height, 1), width=max(width, 1), gsd_m=gsd_m)


def working_gsd(gsd_source: float, gsd_reference: float, k: float = 1.0) -> float:
    """gsd_work = max(gsd_source, gsd_reference) * k.

    The maximum, never the minimum. Working at the finer GSD would require upsampling
    the coarser image, which is the failure mode this pipeline exists to avoid.
    """
    return float(max(gsd_source, gsd_reference) * k)


def prefilter_for_downsample(image: np.ndarray, factor: float) -> np.ndarray:
    """Area-average before a large downsample, which is the correct anti-aliased operator.

    `cv2.INTER_AREA` does this in one step for integer-ish factors; beyond about 4x it is
    applied in stages, because a single enormous area average and a sequence of smaller
    ones differ noticeably at the extreme ratios this pipeline is asked to handle.
    """
    img = np.asarray(image, dtype=np.float32)
    while factor > 4.0:
        img = cv2.resize(img, (max(1, img.shape[1] // 2), max(1, img.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        factor /= 2.0
    return img


def orthorectify(
    product: Product,
    model: SensorModel,
    grid: OrthoGrid,
    dem: np.ndarray | None = None,
    dem_gsd_m: float | None = None,
    apply_parallax: bool = True,
    emission_azimuth_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a product onto the common grid. Returns (ortho, validity mask)."""
    map_x, map_y = grid.map_coords()

    if apply_parallax and dem is not None and abs(product.illum.emission_deg) > 1e-3:
        height = _sample_dem_on_grid(dem, dem_gsd_m or grid.gsd_m, grid, map_x, map_y)
        dx, dy = parallax_shift(height, product.illum.emission_deg, emission_azimuth_deg)
        map_x = map_x + dx
        map_y = map_y + dy
        parallax_applied = True
    else:
        parallax_applied = False
    model.parallax_applied = parallax_applied

    pts = np.column_stack([map_x.ravel(), map_y.ravel()])
    img_pts = model.map_to_image(pts)
    xs = img_pts[:, 0].reshape(grid.height, grid.width).astype(np.float32)
    ys = img_pts[:, 1].reshape(grid.height, grid.width).astype(np.float32)

    source = product.pan()
    # Estimate how much the source is being decimated, and pre-filter if it is a lot.
    step = max(1, grid.height // 8)
    dx_img = np.abs(np.diff(xs[::step, ::step], axis=1)).mean() if xs.shape[1] > step else 1.0
    dy_img = np.abs(np.diff(ys[::step, ::step], axis=0)).mean() if xs.shape[0] > step else 1.0
    factor = float(max(dx_img, dy_img, 1.0))
    if factor > 1.5:
        pre = prefilter_for_downsample(source, factor)
        sx = pre.shape[1] / source.shape[1]
        sy = pre.shape[0] / source.shape[0]
        source, xs, ys = pre, xs * sx, ys * sy

    ortho = cv2.remap(
        np.ascontiguousarray(source, dtype=np.float32), xs, ys,
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan,
    )
    mask = np.isfinite(ortho) & (xs >= 0) & (ys >= 0) & (xs < source.shape[1]) & (ys < source.shape[0])
    return np.nan_to_num(ortho, nan=0.0).astype(np.float32), mask


def _sample_dem_on_grid(
    dem: np.ndarray,
    dem_gsd_m: float,
    grid: OrthoGrid,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """DEM height above the local mean, sampled on the output grid.

    Referencing to the patch mean rather than the datum matters: the constant part of the
    terrain height is already absorbed by the sensor model's corner fit, so adding it
    again through the parallax term would double-count it.
    """
    d = np.asarray(dem, dtype=np.float32)
    scale = dem_gsd_m / grid.gsd_m if dem_gsd_m else 1.0
    resized = cv2.resize(d, (grid.width, grid.height), interpolation=cv2.INTER_LINEAR) if scale != 1.0 or d.shape != (grid.height, grid.width) else d
    return (resized - float(np.nanmean(resized))).astype(np.float32)


def footprint_bounds_map(product: Product, crs: CRS) -> tuple[float, float, float, float]:
    """Map-coordinate bounds of a product's footprint."""
    lon, lat = np.asarray(product.footprint.exterior.coords).T
    x, y = lonlat_to_map(crs, lon, lat)
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def intersect_bounds(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Intersection of two map-coordinate bounding boxes, or None if they are disjoint."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def grid_to_lonlat(grid: OrthoGrid, pts_px: np.ndarray) -> np.ndarray:
    """Grid pixel coordinates to selenographic lon/lat, for the tie-point file."""
    p = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
    x = grid.transform.c + (p[:, 0] + 0.5) * grid.transform.a
    y = grid.transform.f + (p[:, 1] + 0.5) * grid.transform.e
    lon, lat = map_to_lonlat(grid.crs, x, y)
    return np.column_stack([lon, lat])
