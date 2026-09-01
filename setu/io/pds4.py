"""PDS4 reader for Chandrayaan-2 Level-1 OHRC, TMC-2 and IIRS products.

S0 is specific that the XML label is authoritative, not the `.img` header: the ISRO
PDS4 labels carry the corrected selenographic corners and the illumination keywords,
while the detached binary carries only pixels. `pds4_tools` parses the label and
memory-maps the array, which is what keeps a 12,000 x 90,000 OHRC strip openable.
"""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from setu.io.labels import (
    corners_from_label,
    find,
    find_datetime,
    find_float,
    flatten,
    resolve_illumination,
)
from setu.types import NOMINAL_GSD_M, GeometryCube, Product

#: Structure names that carry geometry backplanes rather than radiance.
_BACKPLANE_HINTS = {
    "incidence": "incidence_deg",
    "emission": "emission_deg",
    "phase": "phase_deg",
    "solar_azimuth": "sun_az_deg",
    "sun_azimuth": "sun_az_deg",
    "longitude": "lon_deg",
    "latitude": "lat_deg",
}


def _xml_to_dict(elem: ET.Element) -> Any:
    """Collapse a PDS4 XML subtree to nested dicts, dropping namespaces."""
    tag = elem.tag.split("}")[-1]
    children = list(elem)
    if not children:
        return (elem.text or "").strip()
    out: dict[str, Any] = {}
    for child in children:
        ctag = child.tag.split("}")[-1]
        val = _xml_to_dict(child)
        if ctag in out:
            if not isinstance(out[ctag], list):
                out[ctag] = [out[ctag]]
            out[ctag].append(val)
        else:
            out[ctag] = val
    return out if tag else out


def parse_pds4_label(xml_path: str | Path) -> dict[str, Any]:
    """Parse a PDS4 label into a plain nested dictionary, namespaces stripped."""
    root = ET.parse(str(xml_path)).getroot()
    return {root.tag.split("}")[-1]: _xml_to_dict(root)}


def _sensor_from_label(flat: dict[str, Any], hint: str | None) -> str:
    if hint:
        return hint
    text = " ".join(str(v).upper() for v in flat.values() if isinstance(v, str))
    if "OHRC" in text or "ORBITER HIGH RESOLUTION" in text:
        return "OHRC"
    if "IIRS" in text or "IMAGING INFRARED SPECTROMETER" in text:
        return "IIRS"
    if "TMC" in text:
        for view, name in (("FORE", "TMC2_FORE"), ("AFT", "TMC2_AFT"), ("NADIR", "TMC2_NADIR")):
            if view in text:
                return name
        return "TMC2_NADIR"
    return "SYNTHETIC"


def _wavelengths_from_label(flat: dict[str, Any], n_bands: int) -> np.ndarray | None:
    """IIRS band centres in nanometres.

    The PDS4 label may carry an explicit band array; when it does not, the IIRS
    dispersion is close enough to linear over 0.8-5.0 um to reconstruct centres from
    the band count, which is all the pseudo-panchromatic window selection needs.
    """
    for key in ("band_wavelength", "center_wavelength", "wavelength"):
        v = find(flat, key)
        if isinstance(v, (list, tuple)) and len(v) == n_bands:
            return np.asarray([float(x) for x in v], dtype=np.float64)
    if n_bands > 32:
        return np.linspace(800.0, 5000.0, n_bands)
    return None


def read_pds4(xml_path: str | Path, sensor: str | None = None, lazy: bool = True) -> Product:
    """Read one PDS4 product. `lazy` keeps OHRC-sized arrays memory-mapped."""
    import pds4_tools

    xml_path = Path(xml_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        struct_list = pds4_tools.read(str(xml_path), quiet=True, lazy_load=lazy)

    label = parse_pds4_label(xml_path)
    flat = flatten(label)

    image = None
    geom_arrays: dict[str, np.ndarray] = {}
    for s in struct_list:
        name = (getattr(s, "id", "") or "").lower()
        matched = next((f for hint, f in _BACKPLANE_HINTS.items() if hint in name), None)
        if matched is not None:
            geom_arrays[matched] = np.asarray(s.data, dtype=np.float32)
        elif image is None and getattr(s, "is_array", lambda: False)():
            image = s

    if image is None:
        arrays = [s for s in struct_list if getattr(s, "is_array", lambda: False)()]
        if not arrays:
            raise ValueError(f"{xml_path}: no array structure found in the PDS4 label")
        image = max(arrays, key=lambda s: np.prod(np.asarray(s.data).shape))

    arr = np.asarray(image.data)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))     # band-sequential to (h, w, bands)
    arr = arr.astype(np.float32, copy=False)

    sensor_name = _sensor_from_label(flat, sensor)
    geometry = GeometryCube(**geom_arrays) if geom_arrays else None
    illum = resolve_illumination(flat, geometry=geometry, pid=xml_path.stem)

    corners = corners_from_label(flat)
    footprint = _footprint(corners, flat)
    gsd = find_float(flat, "gsd") or NOMINAL_GSD_M.get(sensor_name, 1.0)
    n_bands = arr.shape[2] if arr.ndim == 3 else 1

    return Product(
        pid=str(find(flat, "product_id") or xml_path.stem),
        sensor=sensor_name,                                  # type: ignore[arg-type]
        array=arr,
        gsd_m=float(gsd),
        footprint=footprint,
        corner_latlon=corners if corners is not None else _corners_from_polygon(footprint),
        illum=illum,
        acquisition_utc=find_datetime(flat),
        band_wavelengths_nm=_wavelengths_from_label(flat, n_bands) if n_bands > 1 else None,
        geometry=geometry,
        label=label,
        path=str(xml_path),
    )


def _footprint(corners: np.ndarray | None, flat: dict[str, Any]) -> Polygon:
    """Footprint polygon in selenographic lon/lat."""
    if corners is not None:
        return Polygon([(lon, lat) for lat, lon in corners]).buffer(0)
    lon = find_float(flat, "center_longitude") or 0.0
    lat = find_float(flat, "center_latitude") or 0.0
    d = 0.05
    return Polygon([(lon - d, lat - d), (lon + d, lat - d), (lon + d, lat + d), (lon - d, lat + d)])


def _corners_from_polygon(poly: Polygon) -> np.ndarray:
    lon0, lat0, lon1, lat1 = poly.bounds
    return np.asarray([[lat1, lon0], [lat1, lon1], [lat0, lon1], [lat0, lon0]], dtype=np.float64)
