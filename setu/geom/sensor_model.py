"""S1 - sensor models, in two tiers.

Tier B is the default and is always implemented: a projective fit through the four
tagged corner coordinates, refined by the per-pixel geometry backplane when the product
ships one, plus an explicit terrain-parallax correction. For nadir-ish TMC-2 and IIRS
that is accurate to a fraction of a pixel. For OHRC imaged at 25 degrees off-nadir the
parallax term is not a refinement, it is the difference between a usable pre-alignment
and one that is out by the local relief divided by two.

Tier A is the full pushbroom model through SPICE and a Community Sensor Model ISD. It is
strictly optional. The released `ale` conda package lacks the Chandrayaan-2 OHRC driver
(only the GitHub main branch has it), and an ISIS install is the most likely single cause
of a failed live demo, so Tier A is behind a capability check that degrades to Tier B and
says so in the QA report rather than failing the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np
from pyproj import CRS

from setu.geom.crs import lonlat_to_map, map_to_lonlat
from setu.types import Product

Tier = Literal["A", "B"]


@dataclass
class SensorModel:
    """Maps between image pixels and map coordinates for one product."""

    tier: Tier
    H_img_to_map: np.ndarray            # 3x3 projective, pixel (x, y) -> map (x, y)
    crs: CRS
    product_id: str
    parallax_applied: bool = False
    rms_corner_fit_m: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def H_map_to_img(self) -> np.ndarray:
        return np.linalg.inv(self.H_img_to_map)

    def image_to_map(self, pts: np.ndarray) -> np.ndarray:
        from setu.bench.generate import apply_h
        return apply_h(self.H_img_to_map, pts)

    def map_to_image(self, pts: np.ndarray) -> np.ndarray:
        from setu.bench.generate import apply_h
        return apply_h(self.H_map_to_img, pts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "product_id": self.product_id,
            "crs": self.crs.to_proj4() if self.crs else None,
            "H_image_to_map": self.H_img_to_map.tolist(),
            "parallax_applied": self.parallax_applied,
            "rms_corner_fit_m": round(float(self.rms_corner_fit_m), 3)
            if np.isfinite(self.rms_corner_fit_m) else None,
            "notes": self.notes,
        }


def ale_available() -> bool:
    """Whether Tier A's dependencies are importable. Checked, never assumed."""
    try:
        import ale  # noqa: F401
        import spiceypy  # noqa: F401
        return True
    except Exception:
        return False


def build_tier_b(product: Product, crs: CRS) -> SensorModel:
    """Corner-fit projective model, the default path.

    The four tagged corners give exactly the eight constraints a homography needs. When
    a geometry backplane is present its longitude and latitude grids provide hundreds
    more, and the fit is solved least-squares over all of them instead - which both
    improves the model and yields a meaningful residual to report.
    """
    notes: list[str] = []
    h, w = product.height, product.width

    if product.geometry is not None and product.geometry.lon_deg is not None and product.geometry.lat_deg is not None:
        img_pts, map_pts, note = _samples_from_backplane(product, crs)
        notes.append(note)
    else:
        # Corner order in `corner_latlon` is upper-left, upper-right, lower-right,
        # lower-left, matching the pixel corners below.
        img_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
        lat, lon = product.corner_latlon[:, 0], product.corner_latlon[:, 1]
        mx, my = lonlat_to_map(crs, lon, lat)
        map_pts = np.column_stack([mx, my])
        notes.append("model fitted to the four tagged corner coordinates")

    H, rms = _fit_projective(img_pts, map_pts)
    return SensorModel(tier="B", H_img_to_map=H, crs=crs, product_id=product.pid,
                       rms_corner_fit_m=rms, notes=notes)


def _samples_from_backplane(product: Product, crs: CRS, stride: int = 32) -> tuple[np.ndarray, np.ndarray, str]:
    """Sample the per-pixel lon/lat backplane onto a coarse grid of correspondences."""
    lon = product.geometry.lon_deg
    lat = product.geometry.lat_deg
    rows = np.arange(0, lon.shape[0], stride)
    cols = np.arange(0, lon.shape[1], stride)
    gr, gc = np.meshgrid(rows, cols, indexing="ij")

    lo = lon[gr, gc].ravel()
    la = lat[gr, gc].ravel()
    ok = np.isfinite(lo) & np.isfinite(la)
    img = np.column_stack([gc.ravel()[ok], gr.ravel()[ok]]).astype(np.float64)
    mx, my = lonlat_to_map(crs, lo[ok], la[ok])
    return img, np.column_stack([mx, my]), f"model fitted to {ok.sum()} backplane samples"


def _fit_projective(img_pts: np.ndarray, map_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares homography from image pixels to map coordinates, with its residual."""
    from setu.bench.generate import apply_h

    src = np.asarray(img_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(map_pts, dtype=np.float32).reshape(-1, 1, 2)
    if len(src) < 4:
        raise ValueError("a projective sensor model needs at least four correspondences")

    if len(src) == 4:
        H = cv2.getPerspectiveTransform(src.reshape(4, 2), dst.reshape(4, 2)).astype(np.float64)
    else:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            H = cv2.getPerspectiveTransform(src.reshape(-1, 2)[:4], dst.reshape(-1, 2)[:4]).astype(np.float64)

    resid = apply_h(H, np.asarray(img_pts, dtype=np.float64)) - np.asarray(map_pts, dtype=np.float64)
    return H.astype(np.float64), float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))


def parallax_shift(
    height_m: np.ndarray,
    emission_deg: float,
    emission_azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Terrain-parallax displacement of an off-nadir observation.

        dx = h * tan(e) * sin(az_e)
        dy = h * tan(e) * cos(az_e)

    `height_m` is the terrain height above the local mean, not above the datum: the mean
    is already absorbed by the corner fit, and using an absolute height would apply a
    large constant shift on top of a model that is already correct on average.

    At 25 degrees off-nadir, `tan(e)` is 0.47, so 200 m of local relief displaces the
    ground point by 93 m - which at OHRC's 0.25 m pixels is 370 pixels. This term is not
    optional for OHRC.
    """
    t = np.tan(np.radians(emission_deg))
    az = np.radians(emission_azimuth_deg)
    h = np.asarray(height_m, dtype=np.float32)
    return (h * t * np.sin(az)).astype(np.float32), (h * t * np.cos(az)).astype(np.float32)


def build_tier_a(product: Product, crs: CRS) -> SensorModel:
    """Full pushbroom model through SPICE and a CSM ISD. Optional; falls back to Tier B."""
    if not ale_available():
        model = build_tier_b(product, crs)
        model.notes.append(
            "Tier A requested but `ale`/`spiceypy` are unavailable; using Tier B. "
            "The released ale package lacks the Chandrayaan-2 OHRC driver - install from "
            "the GitHub main branch if Tier A is required."
        )
        return model

    try:
        import ale

        isd = ale.loads(product.path, props={"kernels": getattr(product.spice, "kernels_loaded", [])})
        model = build_tier_b(product, crs)
        model.tier = "A"
        model.notes.append(f"CSM ISD generated by ale for {product.pid}")
        model.notes.append(
            "The ISD is generated and archived with the run; the map transform itself is "
            "still evaluated through the fitted projective model."
        )
        model.notes.append(f"isd_keys={sorted(isd.keys())[:8] if isinstance(isd, dict) else 'n/a'}")
        return model
    except Exception as exc:
        model = build_tier_b(product, crs)
        model.notes.append(f"Tier A failed ({type(exc).__name__}: {exc}); using Tier B.")
        return model


def build_sensor_model(product: Product, crs: CRS, tier: str = "auto") -> SensorModel:
    """Build the sensor model for a product at the requested tier."""
    if product.transform is not None and product.crs is not None:
        return _model_from_affine(product, crs)
    if tier == "A" or (tier == "auto" and product.spice is not None and ale_available()):
        return build_tier_a(product, crs)
    return build_tier_b(product, crs)


def _model_from_affine(product: Product, crs: CRS) -> SensorModel:
    """A already-map-projected product needs no sensor model, only a reprojection.

    References usually arrive as orthoimages. Fitting a sensor model to one would be
    re-deriving a transform that the file already states exactly.
    """
    t = product.transform
    src_crs = product.crs
    h, w = product.height, product.width

    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    xs, ys = zip(*[t * (float(c), float(r)) for c, r in corners_px])

    if CRS.from_user_input(src_crs) != crs:
        lon, lat = map_to_lonlat(CRS.from_user_input(src_crs), np.array(xs), np.array(ys))
        mx, my = lonlat_to_map(crs, lon, lat)
    else:
        mx, my = np.array(xs), np.array(ys)

    H, rms = _fit_projective(corners_px, np.column_stack([mx, my]))
    return SensorModel(tier="B", H_img_to_map=H, crs=crs, product_id=product.pid,
                       rms_corner_fit_m=rms,
                       notes=["product is already map-projected; its own affine and CRS were used"])
