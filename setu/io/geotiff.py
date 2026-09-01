"""GeoTIFF reader and writer for reference mosaics, SLDEM tiles and derived products.

Reference data is almost always already map-projected, so this reader keeps the
affine and CRS on the `Product` and lets S1 skip straight to resampling rather than
inventing a sensor model for an orthoimage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine, array_bounds
from rasterio.windows import Window
from shapely.geometry import Polygon

from setu.geom.crs import map_to_lonlat, moon_geographic_crs
from setu.io.labels import illumination_from_label
from setu.types import IlluminationState, NOMINAL_GSD_M, Product


def read_geotiff(
    path: str | Path,
    sensor: str = "NAC_L",
    illum: IlluminationState | None = None,
    band: int | None = None,
    lazy: bool = True,
) -> Product:
    """Read a map-projected raster into a `Product`.

    Reference orthoimages rarely carry an illumination state in their tags. When one
    is absent the caller must supply it, because S2's re-illumination needs the
    *source* geometry, not the reference's, and a wrong reference sun angle only ever
    shows up as a quietly worse inlier ratio.
    """
    path = Path(path)
    with rasterio.open(path) as ds:
        tags = {k.lower(): v for k, v in ds.tags().items()}
        crs = ds.crs
        transform = ds.transform
        gsd = float(abs(transform.a))

        if lazy and ds.width * ds.height > 64_000_000:
            arr = np.empty((0, 0), dtype=np.float32)     # window-only access
            h, w = ds.height, ds.width
        else:
            arr = ds.read(band or 1).astype(np.float32) if (band or ds.count == 1) else np.transpose(
                ds.read().astype(np.float32), (1, 2, 0)
            )
            h, w = ds.height, ds.width

        nodata = ds.nodata
        bounds = array_bounds(h, w, transform)

    footprint = _bounds_to_footprint(bounds, crs)
    state = illum or illumination_from_label(tags) or IlluminationState(
        sun_az_deg=0.0, sun_elev_deg=45.0, source="label"
    )

    def _window(r0: int, c0: int, hh: int, ww: int) -> np.ndarray:
        with rasterio.open(path) as d:
            a = d.read(band or 1, window=Window(c0, r0, ww, hh), boundless=True, fill_value=0)
        return a.astype(np.float32)

    if arr.size == 0:
        arr = np.zeros((h, w), dtype=np.float32)          # shape carrier only

    prod = Product(
        pid=path.stem,
        sensor=sensor,                                     # type: ignore[arg-type]
        array=arr,
        gsd_m=gsd if gsd > 0 else NOMINAL_GSD_M.get(sensor, 1.0),
        footprint=footprint,
        corner_latlon=_footprint_corners(footprint),
        illum=state,
        label={"tags": tags, "nodata": nodata, "crs": str(crs)},
        path=str(path),
        transform=transform,
        crs=crs,
    )
    prod._lazy_reader = _window
    return prod


def _bounds_to_footprint(bounds: tuple[float, float, float, float], crs: Any) -> Polygon:
    """Projected bounds to a lon/lat footprint on the Moon sphere."""
    x0, y0, x1, y1 = bounds
    xs = np.array([x0, x1, x1, x0])
    ys = np.array([y1, y1, y0, y0])
    if crs is None or crs.is_geographic:
        lon, lat = xs, ys
    else:
        lon, lat = map_to_lonlat(crs, xs, ys)
    return Polygon(zip(lon, lat)).buffer(0)


def _footprint_corners(poly: Polygon) -> np.ndarray:
    lon0, lat0, lon1, lat1 = poly.bounds
    return np.asarray([[lat1, lon0], [lat1, lon1], [lat0, lon1], [lat0, lon0]], dtype=np.float64)


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    transform: Affine,
    crs: Any,
    nodata: float | None = 0.0,
    cog: bool = True,
    dtype: str | None = None,
) -> Path:
    """Write a raster, tiled and overviewed so it reads as a cloud-optimised GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    out_dtype = dtype or ("float32" if arr.dtype.kind == "f" else str(arr.dtype))

    profile: dict[str, Any] = dict(
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=out_dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=3 if out_dtype.startswith("float") else 2,
    )
    if cog:
        profile.update(tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")

    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr.astype(out_dtype))
        if cog:
            factors = [f for f in (2, 4, 8, 16) if min(arr.shape[1:]) // f >= 64]
            if factors:
                ds.build_overviews(factors, rasterio.enums.Resampling.average)
                ds.update_tags(ns="rio_overview", resampling="average")
    return path


def geographic_transform(lon0: float, lat1: float, gsd_deg: float) -> Affine:
    """Convenience affine for a lon/lat grid with north up."""
    return Affine(gsd_deg, 0.0, lon0, 0.0, -gsd_deg, lat1)


def moon_crs_for_writing(crs: Any | None) -> Any:
    return crs if crs is not None else moon_geographic_crs()
