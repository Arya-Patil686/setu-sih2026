"""PDS3 reader for LRO NAC EDR/CDR and other older archives.

Implemented with `pvl` plus `numpy.memmap` exactly as S0 specifies. A NAC EDR is a
detached or attached label followed by a raw raster, and memory-mapping it means the
reader costs nothing until a window is actually requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from setu.io.labels import (
    corners_from_label,
    find,
    find_datetime,
    find_float,
    find_int,
    flatten,
    resolve_illumination,
)
from setu.types import NOMINAL_GSD_M, Product

#: PDS3 SAMPLE_TYPE to numpy dtype. The sign of the byte order matters: LRO products
#: are LSB, most ISRO derived products are MSB, and getting this wrong produces an
#: image that looks like noise rather than an error.
_DTYPES: dict[tuple[str, int], str] = {
    ("lsb_integer", 8): "<i1", ("lsb_integer", 16): "<i2", ("lsb_integer", 32): "<i4",
    ("msb_integer", 8): ">i1", ("msb_integer", 16): ">i2", ("msb_integer", 32): ">i4",
    ("lsb_unsigned_integer", 8): "<u1", ("lsb_unsigned_integer", 16): "<u2", ("lsb_unsigned_integer", 32): "<u4",
    ("msb_unsigned_integer", 8): ">u1", ("msb_unsigned_integer", 16): ">u2", ("msb_unsigned_integer", 32): ">u4",
    ("unsigned_integer", 8): "|u1", ("unsigned_integer", 16): "<u2",
    ("integer", 8): "|i1", ("integer", 16): "<i2",
    ("ieee_real", 32): ">f4", ("ieee_real", 64): ">f8",
    ("pc_real", 32): "<f4", ("pc_real", 64): "<f8",
    ("real", 32): "<f4",
}


def _dtype_of(flat: dict[str, Any]) -> str:
    st = str(find(flat, "sample_type") or "lsb_unsigned_integer").strip().lower()
    bits = find_int(flat, "sample_bits") or 8
    if (st, bits) in _DTYPES:
        return _DTYPES[(st, bits)]
    signed = "unsigned" not in st
    endian = ">" if st.startswith("msb") else "<"
    if "real" in st:
        return f"{endian}f{max(4, bits // 8)}"
    return f"{endian}{'i' if signed else 'u'}{max(1, bits // 8)}"


def _data_path(lbl_path: Path, flat: dict[str, Any]) -> tuple[Path, int]:
    """Locate the raster and its byte offset, handling attached and detached labels."""
    pointer = find(flat, "^image") or find(flat, "^qube") or find(flat, "image_pointer")
    record_bytes = find_int(flat, "record_bytes") or 1

    if pointer is None:                                  # attached label
        return lbl_path, _attached_offset(lbl_path)

    if isinstance(pointer, (list, tuple)) and len(pointer) == 2:
        name, rec = str(pointer[0]), int(find_float(pointer[1]) or 1)
        return _sibling(lbl_path, name), (rec - 1) * record_bytes
    if isinstance(pointer, (int, float)):
        return lbl_path, (int(pointer) - 1) * record_bytes
    text = str(pointer).strip().strip("()").strip('"')
    if text.replace(".", "").isdigit():
        return lbl_path, (int(float(text)) - 1) * record_bytes
    return _sibling(lbl_path, text.split(",")[0].strip().strip('"')), 0


def _sibling(lbl_path: Path, name: str) -> Path:
    """Resolve a pointer filename beside the label, tolerating archive case shifts."""
    cand = lbl_path.parent / name
    if cand.exists():
        return cand
    for alt in (name.upper(), name.lower(), lbl_path.stem + Path(name).suffix):
        p = lbl_path.parent / alt
        if p.exists():
            return p
    for p in lbl_path.parent.iterdir():
        if p.stem.lower() == lbl_path.stem.lower() and p.suffix.lower() in (".img", ".dat", ".qub"):
            return p
    raise FileNotFoundError(f"{lbl_path}: cannot locate the raster pointed at by {name!r}")


def _attached_offset(lbl_path: Path) -> int:
    data = lbl_path.read_bytes()
    idx = data.find(b"END\r\n")
    if idx < 0:
        idx = data.find(b"END\n")
    return idx + 5 if idx >= 0 else 0


def read_pds3(lbl_path: str | Path, sensor: str | None = None) -> Product:
    """Read a PDS3 product, memory-mapping the raster."""
    import pvl

    lbl_path = Path(lbl_path)
    label = pvl.load(str(lbl_path))
    flat = flatten(dict(label))

    lines = find_int(flat, "lines")
    samples = find_int(flat, "samples")
    bands = find_int(flat, "bands") or 1
    if not lines or not samples:
        raise ValueError(f"{lbl_path}: label carries no LINES/LINE_SAMPLES")

    path, offset = _data_path(lbl_path, flat)
    dtype = np.dtype(_dtype_of(flat))
    shape = (lines, samples) if bands == 1 else (bands, lines, samples)
    mm = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
    if bands > 1:
        mm = np.transpose(mm, (1, 2, 0))

    name = sensor or _nac_sensor(flat)
    illum = resolve_illumination(flat, pid=lbl_path.stem)
    corners = corners_from_label(flat)
    footprint = (
        Polygon([(lon, lat) for lat, lon in corners]).buffer(0)
        if corners is not None
        else _footprint_from_centre(flat)
    )

    def _window(r0: int, c0: int, h: int, w: int) -> np.ndarray:
        return np.asarray(mm[r0:r0 + h, c0:c0 + w], dtype=np.float32)

    prod = Product(
        pid=str(find(flat, "product_id") or lbl_path.stem),
        sensor=name,                                        # type: ignore[arg-type]
        array=mm,                                           # memmap; never fully materialised
        gsd_m=float(find_float(flat, "gsd") or NOMINAL_GSD_M.get(name, 1.0)),
        footprint=footprint,
        corner_latlon=corners if corners is not None else _corners(footprint),
        illum=illum,
        acquisition_utc=find_datetime(flat),
        label=dict(label),
        path=str(lbl_path),
    )
    prod._lazy_reader = _window
    return prod


def _nac_sensor(flat: dict[str, Any]) -> str:
    inst = str(find(flat, "instrument") or "").upper()
    pid = str(find(flat, "product_id") or "").upper()
    if "NACR" in inst or pid.endswith("RE"):
        return "NAC_R"
    if "NACL" in inst or pid.endswith("LE"):
        return "NAC_L"
    if "WAC" in inst:
        return "WAC"
    return "NAC_L"


def _footprint_from_centre(flat: dict[str, Any]) -> Polygon:
    lon = find_float(flat, "center_longitude") or 0.0
    lat = find_float(flat, "center_latitude") or 0.0
    d = 0.05
    return Polygon([(lon - d, lat - d), (lon + d, lat - d), (lon + d, lat + d), (lon - d, lat + d)])


def _corners(poly: Polygon) -> np.ndarray:
    lon0, lat0, lon1, lat1 = poly.bounds
    return np.asarray([[lat1, lon0], [lat1, lon1], [lat0, lon1], [lat0, lon0]], dtype=np.float64)
