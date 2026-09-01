"""Sensor registry: reader dispatch and the reference-selection policy.

S2c requires the reference choice to be an explicit, inspectable table rather than
something buried in a demo script. Matching 80 m IIRS against 0.5 m NAC at native
resolution is not a sensible task, and a system that knowingly declines to attempt it
is stronger than one that brute-forces a 160x ratio.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from setu.types import Product


@dataclass(frozen=True)
class ReferenceChoice:
    """What a given payload should be registered against, and within what GSD band."""

    preferred: tuple[str, ...]
    acceptable_gsd_m: tuple[float, float]
    rationale: str


#: Payload to reference policy. `acceptable_gsd_m` is the working-GSD band inside which
#: the pairing is defensible; outside it, SETU warns and reports the ratio it was asked
#: to work at rather than silently pretending the result is comparable.
REFERENCE_POLICY: dict[str, ReferenceChoice] = {
    "OHRC": ReferenceChoice(
        preferred=("NAC_L", "NAC_R"),
        acceptable_gsd_m=(0.25, 4.0),
        rationale=(
            "LRO NAC at 0.5-2 m is the only orbital reference within a small factor of "
            "OHRC's 0.25 m. The ratio is 2-8x, which pre-alignment absorbs cleanly."
        ),
    ),
    "TMC2_NADIR": ReferenceChoice(
        preferred=("NAC_L", "NAC_R", "KAGUYA_TC"),
        acceptable_gsd_m=(2.0, 20.0),
        rationale=(
            "TMC-2 at 5 m against NAC is a 10-20x ratio; Kaguya TC at ~10 m is the "
            "gentler partner when a NAC scene with a usable sun angle is unavailable."
        ),
    ),
    "IIRS": ReferenceChoice(
        preferred=("KAGUYA_TC", "SLDEM_HILLSHADE", "WAC"),
        acceptable_gsd_m=(10.0, 120.0),
        rationale=(
            "IIRS pixels are 80 m. The correct reference is Kaguya TC (~10 m), an "
            "SLDEM2015 hillshade (59 m) or a WAC mosaic (~100 m), resampled to ~80 m. "
            "NAC at native resolution is explicitly not attempted."
        ),
    ),
}
REFERENCE_POLICY["TMC2_FORE"] = REFERENCE_POLICY["TMC2_NADIR"]
REFERENCE_POLICY["TMC2_AFT"] = REFERENCE_POLICY["TMC2_NADIR"]
REFERENCE_POLICY["SYNTHETIC"] = ReferenceChoice(
    preferred=("SYNTHETIC",),
    acceptable_gsd_m=(0.01, 1000.0),
    rationale="The controlled benchmark pairs a rendering with another rendering of the same DEM.",
)


def reference_policy(sensor: str) -> ReferenceChoice:
    return REFERENCE_POLICY.get(
        sensor,
        ReferenceChoice(preferred=("NAC_L",), acceptable_gsd_m=(0.1, 200.0), rationale="Default policy."),
    )


def check_pairing(source: Product, reference: Product) -> dict[str, object]:
    """Judge a source/reference pairing against the policy table.

    Returns a verdict rather than raising: an out-of-band pairing is sometimes exactly
    the stress case you want to measure, so long as the report says so.
    """
    pol = reference_policy(source.sensor)
    lo, hi = pol.acceptable_gsd_m
    ratio = max(source.gsd_m, reference.gsd_m) / max(1e-9, min(source.gsd_m, reference.gsd_m))
    in_band = lo <= reference.gsd_m <= hi
    return {
        "source_sensor": source.sensor,
        "reference_sensor": reference.sensor,
        "scale_ratio": round(float(ratio), 3),
        "reference_gsd_m": reference.gsd_m,
        "acceptable_gsd_band_m": [lo, hi],
        "preferred_references": list(pol.preferred),
        "within_policy": bool(in_band and reference.sensor in pol.preferred),
        "within_gsd_band": bool(in_band),
        "rationale": pol.rationale,
    }


# ------------------------------------------------------------------ dispatch

def _reader_pds4(p: Path, **kw) -> Product:
    from setu.io.pds4 import read_pds4
    return read_pds4(p, **kw)


def _reader_pds3(p: Path, **kw) -> Product:
    from setu.io.pds3 import read_pds3
    return read_pds3(p, **kw)


def _reader_geotiff(p: Path, **kw) -> Product:
    from setu.io.geotiff import read_geotiff
    return read_geotiff(p, **kw)


def _reader_isis(p: Path, **kw) -> Product:
    from setu.io.isis import read_isis_cube
    return read_isis_cube(p, **kw)


EXTENSION_READERS: dict[str, Callable[..., Product]] = {
    ".xml": _reader_pds4,
    ".lbl": _reader_pds3,
    ".img": _reader_pds3,
    ".tif": _reader_geotiff,
    ".tiff": _reader_geotiff,
    ".vrt": _reader_geotiff,
    ".jp2": _reader_geotiff,
    ".cub": _reader_isis,
}


def read_product(path: str | Path, **kwargs) -> Product:
    """Open any supported product by extension.

    A `.img` beside a `.lbl` is routed to the label, because the detached label is
    where the geometry lives and the raw raster alone cannot be placed on the Moon.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()

    if suffix == ".img":
        for cand in (path.with_suffix(".lbl"), path.with_suffix(".LBL"), path.with_suffix(".xml")):
            if cand.exists():
                path, suffix = cand, cand.suffix.lower()
                break

    reader = EXTENSION_READERS.get(suffix)
    if reader is None:
        raise ValueError(
            f"{path}: unsupported extension {suffix!r}. "
            f"SETU reads {', '.join(sorted(EXTENSION_READERS))}."
        )
    return reader(path, **kwargs)
