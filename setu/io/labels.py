"""Label-keyword mining shared by the PDS3 and PDS4 readers.

Chandrayaan-2 labels, LRO labels and the various derived archives all spell the same
physical quantity differently. Rather than branching per mission, every reader flattens
its label to a dotted-key dictionary and then asks this module for a quantity by
meaning. Anything that cannot be found is returned as None so that S0's resolution
order can fall through to the next tier and, finally, fail loudly.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import numpy as np

from setu.types import IlluminationState, IllumUnknownError

#: Ordered aliases per quantity. First hit wins, so mission-specific spellings come first.
ALIASES: dict[str, tuple[str, ...]] = {
    "sun_azimuth": ("sun_azimuth", "solar_azimuth", "sub_solar_azimuth", "sun_az", "solar_azimuth_angle"),
    "sun_elevation": ("sun_elevation", "solar_elevation", "sun_elev", "solar_elevation_angle"),
    "incidence": ("incidence_angle", "solar_incidence_angle", "incidence"),
    "emission": ("emission_angle", "emmission_angle", "emission"),
    "phase": ("phase_angle", "phase"),
    "subsolar_lon": ("sub_solar_longitude", "subsolar_longitude"),
    "subsolar_lat": ("sub_solar_latitude", "subsolar_latitude"),
    "start_time": ("start_time", "start_date_time", "image_time", "observation_time", "product_creation_time"),
    "product_id": ("product_id", "product_name", "logical_identifier", "image_id", "file_name"),
    "instrument": ("instrument_id", "instrument_name", "instrument_host_id"),
    "lines": ("lines", "image_lines", "line_samples_total", "number_of_lines"),
    "samples": ("line_samples", "samples", "image_line_samples", "number_of_samples"),
    "bands": ("bands", "band_count", "number_of_bands"),
    "sample_bits": ("sample_bits", "bits_per_sample", "sample_bit_mask"),
    "sample_type": ("sample_type", "data_type"),
    "record_bytes": ("record_bytes",),
    "gsd": ("pixel_resolution", "map_scale", "spatial_resolution", "ground_sampling_distance", "resolution"),
    "ul_lat": ("upper_left_latitude", "ul_corner_latitude", "corner_1_latitude", "north_west_latitude"),
    "ul_lon": ("upper_left_longitude", "ul_corner_longitude", "corner_1_longitude", "north_west_longitude"),
    "ur_lat": ("upper_right_latitude", "ur_corner_latitude", "corner_2_latitude", "north_east_latitude"),
    "ur_lon": ("upper_right_longitude", "ur_corner_longitude", "corner_2_longitude", "north_east_longitude"),
    "lr_lat": ("lower_right_latitude", "lr_corner_latitude", "corner_3_latitude", "south_east_latitude"),
    "lr_lon": ("lower_right_longitude", "lr_corner_longitude", "corner_3_longitude", "south_east_longitude"),
    "ll_lat": ("lower_left_latitude", "ll_corner_latitude", "corner_4_latitude", "south_west_latitude"),
    "ll_lon": ("lower_left_longitude", "ll_corner_longitude", "corner_4_longitude", "south_west_longitude"),
}

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested label (PVL group, PDS4 XML tree, dict) to dotted keys."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        items: Iterable = obj.items()
    elif hasattr(obj, "items"):
        items = obj.items()
    else:
        return {prefix.rstrip("."): obj} if prefix else {}
    for k, v in items:
        key = f"{prefix}{k}".lower()
        if isinstance(v, dict) or hasattr(v, "items"):
            out.update(flatten(v, prefix=f"{key}."))
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (dict,)):
            for i, e in enumerate(v):
                out.update(flatten(e, prefix=f"{key}.{i}."))
        else:
            out[key] = v
    return out


def find(flat: dict[str, Any], quantity: str) -> Any:
    """Look up one quantity by meaning, matching on the trailing part of any key."""
    for alias in ALIASES.get(quantity, (quantity,)):
        if alias in flat:
            return flat[alias]
        for key, val in flat.items():
            tail = key.rsplit(".", 1)[-1]
            if tail == alias:
                return val
    return None


def find_float(flat: dict[str, Any], quantity: str) -> float | None:
    """As `find`, coerced to a float, tolerating `12.3 <deg>` style units."""
    v = find(flat, quantity)
    return as_float(v)


def as_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return f if np.isfinite(f) else None
    if hasattr(v, "value"):           # pvl Quantity
        try:
            return float(v.value)
        except Exception:
            pass
    m = _NUM.search(str(v))
    return float(m.group()) if m else None


def find_int(flat: dict[str, Any], quantity: str) -> int | None:
    f = find_float(flat, quantity)
    return int(round(f)) if f is not None else None


def find_datetime(flat: dict[str, Any], quantity: str = "start_time") -> datetime | None:
    v = find(flat, quantity)
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    s = str(v).strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def illumination_from_label(flat: dict[str, Any]) -> IlluminationState | None:
    """Tier 3 of the S0 resolution order: label keywords.

    Some archives carry incidence but not elevation, and the two are complements, so
    either one is enough. Without an azimuth the state is useless for re-illumination
    and None is returned so the caller can fail loudly.
    """
    az = find_float(flat, "sun_azimuth")
    elev = find_float(flat, "sun_elevation")
    inc = find_float(flat, "incidence")
    if elev is None and inc is not None:
        elev = 90.0 - inc
    if az is None or elev is None:
        return None
    return IlluminationState(
        sun_az_deg=az,
        sun_elev_deg=elev,
        emission_deg=find_float(flat, "emission") or 0.0,
        phase_deg=find_float(flat, "phase"),
        incidence_deg=inc,
        subsolar_lon_deg=find_float(flat, "subsolar_lon"),
        subsolar_lat_deg=find_float(flat, "subsolar_lat"),
        source="label",
    )


def corners_from_label(flat: dict[str, Any]) -> np.ndarray | None:
    """The four tagged corner coordinates as a 4 x 2 array of (lat, lon)."""
    keys = (("ul_lat", "ul_lon"), ("ur_lat", "ur_lon"), ("lr_lat", "lr_lon"), ("ll_lat", "ll_lon"))
    pts = []
    for klat, klon in keys:
        lat, lon = find_float(flat, klat), find_float(flat, klon)
        if lat is None or lon is None:
            return None
        pts.append((lat, ((lon + 180.0) % 360.0) - 180.0))
    return np.asarray(pts, dtype=np.float64)


def resolve_illumination(
    flat: dict[str, Any],
    geometry: Any | None = None,
    spice_fn: Any | None = None,
    pid: str = "<unknown>",
) -> IlluminationState:
    """The S0 resolution order, in the order the specification fixes it.

    Backplane, then SPICE, then label keywords, then fail. Never guess: an invented
    sun angle would silently corrupt the re-illumination that the whole system rests on.
    """
    if geometry is not None:
        st = geometry.centre_illumination()
        if st is not None:
            return st
    if spice_fn is not None:
        try:
            st = spice_fn()
            if st is not None:
                return st
        except Exception:
            pass
    st = illumination_from_label(flat)
    if st is not None:
        return st
    raise IllumUnknownError(
        f"{pid}: no illumination geometry available. Tried the per-pixel backplane, SPICE, "
        f"and label keywords ({', '.join(ALIASES['sun_azimuth'][:3])}, ...). "
        "SETU will not guess a sun angle - supply SPICE kernels or a geometry backplane."
    )
