"""Lunar coordinate reference systems.

S1 step 1 requires an explicit PROJ string built on the Moon sphere of radius
1,737,400 m. EPSG lookups are deliberately avoided: the lunar EPSG entries are
incomplete and inconsistent between PROJ releases, and a silently wrong datum here
would show up as a systematic metres-level bias in every reported RMSE.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

from setu.types import MOON_RADIUS_M

Projection = Literal["equirectangular", "polar_stereographic", "oblique_stereographic"]

#: Geographic (lon/lat) CRS on the lunar sphere. Every footprint is stored in this.
MOON_GEOGRAPHIC = (
    f"+proj=longlat +R={MOON_RADIUS_M:.1f} +no_defs"
)


def moon_geographic_crs() -> CRS:
    return CRS.from_proj4(MOON_GEOGRAPHIC)


def equirectangular(lat_ts_deg: float = 0.0, lon_0_deg: float = 0.0) -> CRS:
    """Equirectangular, the default for footprints away from the poles.

    `lat_ts` is the latitude of true scale. Setting it to the scene centre latitude
    keeps the east-west scale honest over the footprint instead of stretching it by
    1/cos(lat), which matters once a scene sits at 60 degrees.
    """
    return CRS.from_proj4(
        f"+proj=eqc +lat_ts={lat_ts_deg:.8f} +lat_0=0 +lon_0={lon_0_deg:.8f} "
        f"+x_0=0 +y_0=0 +R={MOON_RADIUS_M:.1f} +units=m +no_defs"
    )


def polar_stereographic(north: bool) -> CRS:
    """Polar stereographic, used above 60 degrees latitude."""
    lat_0 = 90.0 if north else -90.0
    return CRS.from_proj4(
        f"+proj=stere +lat_0={lat_0} +lat_ts={lat_0} +lon_0=0 "
        f"+k=1 +x_0=0 +y_0=0 +R={MOON_RADIUS_M:.1f} +units=m +no_defs"
    )


def oblique_stereographic(lon_0_deg: float, lat_0_deg: float) -> CRS:
    """Oblique stereographic centred on the scene, for large mid-latitude footprints."""
    return CRS.from_proj4(
        f"+proj=sterea +lat_0={lat_0_deg:.8f} +lon_0={lon_0_deg:.8f} "
        f"+k=1 +x_0=0 +y_0=0 +R={MOON_RADIUS_M:.1f} +units=m +no_defs"
    )


def choose_crs(
    footprint: Polygon,
    polar_lat_deg: float = 60.0,
    oblique_extent_km: float = 50.0,
) -> tuple[CRS, Projection]:
    """Pick the working projection for a footprint, following S1 step 1."""
    lon_c, lat_c = footprint.centroid.x, footprint.centroid.y
    if abs(lat_c) > polar_lat_deg:
        return polar_stereographic(north=lat_c > 0), "polar_stereographic"

    extent_km = footprint_extent_km(footprint)
    if extent_km > oblique_extent_km:
        return oblique_stereographic(lon_c, lat_c), "oblique_stereographic"
    return equirectangular(lat_ts_deg=lat_c, lon_0_deg=lon_c), "equirectangular"


def footprint_extent_km(footprint: Polygon) -> float:
    """Great-circle extent of a footprint's bounding box, in kilometres."""
    lon0, lat0, lon1, lat1 = footprint.bounds
    dlat_km = np.radians(lat1 - lat0) * MOON_RADIUS_M / 1000.0
    dlon_km = np.radians(lon1 - lon0) * MOON_RADIUS_M / 1000.0 * np.cos(np.radians((lat0 + lat1) / 2))
    return float(max(abs(dlat_km), abs(dlon_km)))


def metres_per_degree(lat_deg: float) -> tuple[float, float]:
    """Metres per degree of longitude and of latitude at a given latitude."""
    m_lat = np.radians(1.0) * MOON_RADIUS_M
    m_lon = m_lat * np.cos(np.radians(lat_deg))
    return float(m_lon), float(m_lat)


def make_transformer(src: CRS, dst: CRS) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def lonlat_to_map(crs: CRS, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tr = make_transformer(moon_geographic_crs(), crs)
    x, y = tr.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def map_to_lonlat(crs: CRS, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tr = make_transformer(crs, moon_geographic_crs())
    lon, lat = tr.transform(np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lon), np.asarray(lat)
