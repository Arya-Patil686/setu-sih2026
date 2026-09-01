"""Core data types shared by every SETU stage.

Section 5/S0 of the specification requires one abstraction that every payload and
every reference satisfies, so that nothing downstream ever hard-codes a sensor.
`Product` is that abstraction: OHRC, TMC-2, IIRS, LRO NAC, Kaguya TC, WAC and the
synthetic benchmark renderer all enter the pipeline through it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import numpy as np
from shapely.geometry import Polygon, mapping

# Moon reference sphere used everywhere. The specification pins this value; do not
# substitute an ellipsoid or an EPSG lookup, both of which disagree with the
# planetary community convention for lunar products.
MOON_RADIUS_M = 1_737_400.0

SensorName = Literal[
    "OHRC",
    "TMC2_FORE",
    "TMC2_NADIR",
    "TMC2_AFT",
    "IIRS",
    "NAC_L",
    "NAC_R",
    "KAGUYA_TC",
    "WAC",
    "SLDEM_HILLSHADE",
    "SYNTHETIC",
]

#: Nominal ground sampling distance at 100 km altitude, metres per pixel.
NOMINAL_GSD_M: dict[str, float] = {
    "OHRC": 0.25,
    "TMC2_FORE": 5.0,
    "TMC2_NADIR": 5.0,
    "TMC2_AFT": 5.0,
    "IIRS": 80.0,
    "NAC_L": 1.0,
    "NAC_R": 1.0,
    "KAGUYA_TC": 10.0,
    "WAC": 100.0,
    "SLDEM_HILLSHADE": 59.0,
    "SYNTHETIC": 1.0,
}


class IllumUnknownError(RuntimeError):
    """Raised when illumination geometry cannot be established.

    S0 mandates a strict resolution order for illumination and then `fail loudly`.
    Guessing a sun angle silently poisons every number the eval harness produces,
    so this is an error rather than a warning.
    """


@dataclass
class IlluminationState:
    """Solar and viewing geometry for one acquisition, in degrees.

    `source` records which of the four resolution tiers of S0 supplied the values,
    and is carried all the way into the QA report and the PDS4 label so a reader can
    tell a SPICE-derived angle from a label keyword.
    """

    sun_az_deg: float
    sun_elev_deg: float
    emission_deg: float = 0.0
    phase_deg: float | None = None
    incidence_deg: float | None = None
    subsolar_lon_deg: float | None = None
    subsolar_lat_deg: float | None = None
    source: Literal["backplane", "spice", "label", "synthetic"] = "label"

    def __post_init__(self) -> None:
        self.sun_az_deg = float(self.sun_az_deg) % 360.0
        self.sun_elev_deg = float(self.sun_elev_deg)
        self.emission_deg = float(self.emission_deg)
        if self.incidence_deg is None:
            # Incidence is measured from the local normal, elevation from the horizon.
            self.incidence_deg = 90.0 - self.sun_elev_deg
        if self.phase_deg is None:
            self.phase_deg = self._phase_from_angles()

    def _phase_from_angles(self) -> float:
        """Phase angle from incidence, emission and the relative azimuth.

        With the sun azimuth known but the spacecraft azimuth unknown, the best
        available assumption is a coplanar geometry, which is exact for nadir viewing
        and close for the small emission angles that dominate these payloads.
        """
        i = np.radians(self.incidence_deg or 0.0)
        e = np.radians(self.emission_deg)
        cos_g = np.cos(i) * np.cos(e) + np.sin(i) * np.sin(e)
        return float(np.degrees(np.arccos(np.clip(cos_g, -1.0, 1.0))))

    @property
    def mu0(self) -> float:
        """cos(incidence), clamped at the terminator."""
        return float(max(0.0, np.cos(np.radians(self.incidence_deg or 0.0))))

    @property
    def mu(self) -> float:
        """cos(emission)."""
        return float(max(1e-6, np.cos(np.radians(self.emission_deg))))

    def sun_vector(self) -> np.ndarray:
        """Unit vector towards the Sun in a local East-North-Up frame.

        Azimuth is clockwise from north, which is the planetary convention used by
        both the PDS labels and the SPICE-derived backplanes.
        """
        az = np.radians(self.sun_az_deg)
        el = np.radians(self.sun_elev_deg)
        return np.array(
            [np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeometryCube:
    """Per-pixel geometry backplane, when a product ships one.

    Arrays are all the same shape as the image. Any of them may be None; the
    illumination resolver checks for the ones it needs rather than assuming a
    complete cube.
    """

    incidence_deg: np.ndarray | None = None
    emission_deg: np.ndarray | None = None
    phase_deg: np.ndarray | None = None
    sun_az_deg: np.ndarray | None = None
    lon_deg: np.ndarray | None = None
    lat_deg: np.ndarray | None = None

    def centre_illumination(self) -> IlluminationState | None:
        """Collapse the cube to a single scene-centre illumination state."""
        if self.incidence_deg is None or self.sun_az_deg is None:
            return None
        med = lambda a: float(np.nanmedian(a)) if a is not None else None  # noqa: E731
        inc = med(self.incidence_deg)
        return IlluminationState(
            sun_az_deg=med(self.sun_az_deg) or 0.0,
            sun_elev_deg=90.0 - (inc or 0.0),
            emission_deg=med(self.emission_deg) or 0.0,
            phase_deg=med(self.phase_deg),
            incidence_deg=inc,
            source="backplane",
        )


@dataclass
class SpiceContext:
    """Handle to the mission kernel set, populated only when Tier A is enabled."""

    metakernel: str | None = None
    kernels_loaded: list[str] = field(default_factory=list)
    target: str = "MOON"
    frame: str = "MOON_ME"
    observer: str | None = None
    et: float | None = None


@dataclass
class Product:
    """One image (source or reference) plus everything needed to place it on the Moon."""

    pid: str
    sensor: SensorName
    array: np.ndarray                      # 2-D (pan) or 3-D (h, w, bands), float32
    gsd_m: float                           # nominal, at scene centre
    footprint: Polygon                     # selenographic lon/lat, planetocentric
    corner_latlon: np.ndarray              # 4 x 2 (lat, lon) as tagged in the label
    illum: IlluminationState
    acquisition_utc: datetime | None = None
    band_wavelengths_nm: np.ndarray | None = None
    geometry: GeometryCube | None = None
    spice: SpiceContext | None = None
    label: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    #: Affine transform, if the product is already map-projected (references usually are).
    transform: Any | None = None
    crs: Any | None = None
    #: Set by readers that keep the pixels on disk; see `window`.
    _lazy_reader: Any = field(default=None, repr=False)

    # ---------------------------------------------------------------- geometry

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def n_bands(self) -> int:
        return 1 if self.array.ndim == 2 else int(self.array.shape[2])

    @property
    def is_cube(self) -> bool:
        return self.array.ndim == 3 and self.n_bands > 1

    @property
    def centre_lonlat(self) -> tuple[float, float]:
        c = self.footprint.centroid
        return float(c.x), float(c.y)

    def window(self, row0: int, col0: int, h: int, w: int) -> np.ndarray:
        """Read a window without materialising the whole product.

        OHRC strips are roughly 12,000 x 90,000 px. S0 forbids loading one whole, and
        every downstream stage is written against this method rather than `.array`.
        """
        if self._lazy_reader is not None:
            return self._lazy_reader(row0, col0, h, w)
        row1 = min(row0 + h, self.height)
        col1 = min(col0 + w, self.width)
        return np.ascontiguousarray(self.array[row0:row1, col0:col1])

    def pan(self) -> np.ndarray:
        """A single-band float32 view suitable for matching.

        Multi-band products collapse through the IIRS pseudo-panchromatic synthesis
        rather than a naive mean over all bands.
        """
        if not self.is_cube:
            a = self.array if self.array.ndim == 2 else self.array[:, :, 0]
            return np.asarray(a, dtype=np.float32)
        from setu.illum.iirs import pseudo_panchromatic

        return pseudo_panchromatic(self.array, self.band_wavelengths_nm)

    def summary(self) -> dict[str, Any]:
        lon, lat = self.centre_lonlat
        return {
            "pid": self.pid,
            "sensor": self.sensor,
            "shape": list(self.array.shape),
            "gsd_m": self.gsd_m,
            "centre_lon_deg": round(lon, 6),
            "centre_lat_deg": round(lat, 6),
            "acquisition_utc": self.acquisition_utc.isoformat() if self.acquisition_utc else None,
            "illumination": self.illum.to_dict(),
            "footprint": mapping(self.footprint),
        }


@dataclass
class TiePoint:
    """One correspondence, carrying its own uncertainty.

    The covariance fields are what turn a match list into something an agency can
    ingest, and they are what the adaptive outlier threshold of S5 is derived from.
    """

    tid: int
    src_line: float          # row in the source ortho grid
    src_sample: float        # column in the source ortho grid
    ref_line: float
    ref_sample: float
    conf: float = 0.0
    track: Literal["A", "B", "agreed", "reseed"] = "A"
    sigma_x: float = np.nan
    sigma_y: float = np.nan
    sigma_xy: float = 0.0
    residual_x: float = np.nan
    residual_y: float = np.nan
    inlier: bool = True
    reseeded: bool = False
    cell_id: int = -1
    src_lon: float = np.nan
    src_lat: float = np.nan
    ref_lon: float = np.nan
    ref_lat: float = np.nan

    @property
    def trace_sigma(self) -> float:
        sx, sy = self.sigma_x, self.sigma_y
        if not np.isfinite(sx) or not np.isfinite(sy):
            return np.inf
        return float(sx * sx + sy * sy)

    @property
    def sigma_rms(self) -> float:
        return float(np.sqrt(self.trace_sigma))

    @property
    def residual_norm(self) -> float:
        return float(np.hypot(self.residual_x, self.residual_y))

    @property
    def quality(self) -> float:
        """Ranking score used by the uniformity quota of S6."""
        return float(self.conf / (1.0 + self.trace_sigma))


@dataclass
class RunResult:
    """Everything one `setu register` invocation produced."""

    run_id: str
    source: dict[str, Any]
    reference: dict[str, Any]
    tiepoints: list[TiePoint]
    metrics: dict[str, Any]
    transform: dict[str, Any]
    config: dict[str, Any]
    stages: list[dict[str, Any]] = field(default_factory=list)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def inliers(self) -> list[TiePoint]:
        return [t for t in self.tiepoints if t.inlier]

    def to_json(self, **kw: Any) -> str:
        payload = {
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "source": self.source,
            "reference": self.reference,
            "metrics": self.metrics,
            "transform": self.transform,
            "config": self.config,
            "stages": self.stages,
            "n_tiepoints": len(self.tiepoints),
            "n_inliers": len(self.inliers),
        }
        return json.dumps(payload, indent=2, default=_json_default, **kw)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Polygon):
        return mapping(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)
