"""S1 - geometric pre-alignment, end to end.

This is the stage that spends the geometry. Both products are ortho-projected onto one
map projection at one ground sampling distance using the shape model, which turns two
problems the problem statement names - scale and viewpoint - from appearance problems
into arithmetic. What is left over is a residual planar misalignment of a few hundred
metres to a few kilometres, caused by the kilometre-level error in the Chandrayaan-2
geolocation prior, and that is what the rest of the pipeline is for.

The acceptance test at the end is not decoration. If the residual after pre-alignment is
not a small pure translation, something upstream is wrong - the wrong reference tile, a
mis-parsed corner coordinate, a longitude convention mismatch - and continuing would
produce a confident, wrong answer. It raises instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pyproj import CRS
from skimage.registration import phase_cross_correlation

from setu.geom.crs import choose_crs, footprint_extent_km
from setu.geom.ortho import (
    OrthoGrid,
    prefilter_for_downsample,
    build_grid,
    footprint_bounds_map,
    intersect_bounds,
    orthorectify,
    working_gsd,
)
from setu.geom.sensor_model import SensorModel, build_sensor_model
from setu.types import Product


class PrealignmentError(RuntimeError):
    """Raised when S1's acceptance test fails."""


@dataclass
class PrealignResult:
    """Two co-gridded rasters plus everything needed to reproduce them."""

    src_ortho: np.ndarray
    ref_ortho: np.ndarray
    src_mask: np.ndarray
    ref_mask: np.ndarray
    grid: OrthoGrid
    src_model: SensorModel
    ref_model: SensorModel
    residual_px: float
    residual_shift: tuple[float, float]
    accepted: bool
    meta: dict[str, Any] = field(default_factory=dict)
    #: Maps a point in the working reference frame back to the reference product's own
    #: pixel frame. Identity for a real run, where the working grid *is* the output frame.
    #: On the controlled benchmark's cross-scale pairs the reference is decimated to the
    #: working GSD first, and this carries that factor so the transform SETU reports is
    #: still expressed in the reference product's own pixels.
    frame_correction: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def overlap(self) -> np.ndarray:
        """Validity mask of the common ground, in the *source* frame.

        After a real pre-alignment both rasters share one grid and the overlap is simply
        the intersection. The controlled benchmark's cross-scale pairs are deliberately
        not co-gridded - that is the scale ratio being tested - so the two masks have
        different shapes. There the source footprint lies inside the reference by
        construction, which makes the source's own validity mask the correct answer.
        """
        if self.src_mask.shape == self.ref_mask.shape:
            return self.src_mask & self.ref_mask
        return self.src_mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid.to_dict(),
            "source_model": self.src_model.to_dict(),
            "reference_model": self.ref_model.to_dict(),
            "residual_px": round(float(self.residual_px), 3),
            "residual_shift_px": [round(float(v), 3) for v in self.residual_shift],
            "residual_m": round(float(self.residual_px * self.grid.gsd_m), 2),
            "accepted": self.accepted,
            "overlap_fraction": round(float(self.overlap.mean()), 4),
            **self.meta,
        }


def coarse_residual(
    src: np.ndarray,
    ref: np.ndarray,
    downsample: int = 4,
    mask: np.ndarray | None = None,
) -> tuple[float, tuple[float, float]]:
    """Residual misalignment after pre-alignment, by coarse phase correlation.

    Run on a downsampled pair because the residual is expected to be large - hundreds of
    pixels - and correlating at full resolution would spend most of its time resolving a
    shift the test does not need to know precisely.
    """
    import cv2

    a = np.asarray(src, dtype=np.float32)
    b = np.asarray(ref, dtype=np.float32)
    if mask is not None:
        a = np.where(mask, a, 0.0)
        b = np.where(mask, b, 0.0)

    h = max(a.shape[0] // downsample, 16)
    w = max(a.shape[1] // downsample, 16)
    a_s = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    b_s = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    if a_s.std() < 1e-8 or b_s.std() < 1e-8:
        return float("inf"), (float("nan"), float("nan"))

    # Windowing suppresses the edge discontinuity, which would otherwise pin the peak
    # at zero shift and make every pre-alignment look perfect.
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    a_s = (a_s - a_s.mean()) * win
    b_s = (b_s - b_s.mean()) * win

    try:
        shift, _, _ = phase_cross_correlation(b_s, a_s, upsample_factor=4, normalization="phase")
    except Exception:
        return float("inf"), (float("nan"), float("nan"))

    dy, dx = float(shift[0]) * downsample, float(shift[1]) * downsample
    return float(np.hypot(dx, dy)), (dx, dy)


def prealign(
    source: Product,
    reference: Product,
    dem: np.ndarray | None = None,
    dem_gsd_m: float | None = None,
    gsd_scale_k: float = 1.0,
    polar_lat_deg: float = 60.0,
    oblique_extent_km: float = 50.0,
    apply_parallax: bool = True,
    sensor_model_tier: str = "auto",
    max_size_px: int = 4096,
    max_residual_px: float = 500.0,
    acceptance_downsample: int = 4,
    raise_on_failure: bool = True,
    correction: np.ndarray | None = None,
) -> PrealignResult:
    """Ortho-project both products onto one grid and verify the result.

    `correction` is the S5 -> S1 feedback edge: once a global transform has been
    estimated, the source model is composed with it and pre-alignment is re-run from the
    corrected footprint. One iteration is enough, and the caller enforces the cap.
    """
    combined = source.footprint.union(reference.footprint)
    crs, projection = choose_crs(combined, polar_lat_deg, oblique_extent_km)

    gsd = working_gsd(source.gsd_m, reference.gsd_m, gsd_scale_k)

    bounds_src = footprint_bounds_map(source, crs)
    bounds_ref = footprint_bounds_map(reference, crs)
    bounds = intersect_bounds(bounds_src, bounds_ref)
    if bounds is None:
        raise PrealignmentError(
            f"{source.pid} and {reference.pid} have no overlapping footprint in {projection}. "
            f"Source bounds {bounds_src}, reference bounds {bounds_ref}."
        )

    grid = build_grid(bounds, crs, gsd, max_size_px)

    src_model = build_sensor_model(source, crs, sensor_model_tier)
    ref_model = build_sensor_model(reference, crs, sensor_model_tier)
    if correction is not None:
        # The correction is expressed in map coordinates, so it composes on the left of
        # the image-to-map transform.
        src_model.H_img_to_map = np.asarray(correction, dtype=np.float64) @ src_model.H_img_to_map
        src_model.notes.append("global transform from S5 composed into the sensor model (S5 -> S1 feedback)")

    src_ortho, src_mask = orthorectify(source, src_model, grid, dem, dem_gsd_m, apply_parallax)
    ref_ortho, ref_mask = orthorectify(reference, ref_model, grid, dem, dem_gsd_m, apply_parallax=False)

    overlap = src_mask & ref_mask
    residual_px, shift = coarse_residual(src_ortho, ref_ortho, acceptance_downsample, overlap)
    accepted = bool(np.isfinite(residual_px) and residual_px <= max_residual_px)

    meta = {
        "projection": projection,
        "gsd_source_m": source.gsd_m,
        "gsd_reference_m": reference.gsd_m,
        "gsd_work_m": round(grid.gsd_m, 6),
        "scale_ratio": round(max(source.gsd_m, reference.gsd_m) / max(min(source.gsd_m, reference.gsd_m), 1e-9), 3),
        "footprint_extent_km": round(footprint_extent_km(combined), 3),
        "max_residual_px": max_residual_px,
        "dem_used": dem is not None,
        "parallax_applied": src_model.parallax_applied,
        "emission_difference_deg": round(abs(source.illum.emission_deg - reference.illum.emission_deg), 3),
        "d_sun_elevation_deg": round(abs(source.illum.sun_elev_deg - reference.illum.sun_elev_deg), 3),
        "d_sun_azimuth_deg": round(_angular_diff(source.illum.sun_az_deg, reference.illum.sun_az_deg), 3),
    }

    if not accepted and raise_on_failure:
        raise PrealignmentError(
            f"S1 acceptance test failed for {source.pid} against {reference.pid}: residual "
            f"{residual_px:.1f} px ({residual_px * grid.gsd_m:.0f} m) exceeds the limit of "
            f"{max_residual_px:.0f} px. This usually means a wrong reference tile, a "
            f"longitude-convention mismatch, or corner coordinates that were mis-parsed. "
            f"SETU stops here rather than producing a confident wrong registration."
        )

    return PrealignResult(
        src_ortho=src_ortho, ref_ortho=ref_ortho, src_mask=src_mask, ref_mask=ref_mask,
        grid=grid, src_model=src_model, ref_model=ref_model,
        residual_px=residual_px, residual_shift=shift, accepted=accepted, meta=meta,
    )


def _angular_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def prealign_synthetic(source: Product, reference: Product) -> PrealignResult:
    """Pre-alignment for the controlled benchmark.

    At a scale ratio of one the pair is already co-gridded and this is a pass-through,
    recorded as a real stage with `skipped: true` rather than silently bypassed.

    Above a ratio of one it is not a pass-through, and must not be. The one thing S1 does
    for scale is bring both images to a common ground sampling distance by decimating the
    finer of the two, and skipping that on a cross-scale pair would not be measuring
    SETU - it would be measuring whether a matcher happens to be scale-invariant, which is
    the question the architecture exists to avoid asking. So the same rule applies here as
    everywhere else: work at the coarser GSD, reach it by area-averaging, and never
    upsample the coarser image to meet the finer one.
    """
    import cv2
    from rasterio.transform import Affine

    from setu.geom.crs import equirectangular

    src = source.pan()
    ref = reference.pan()

    gsd_work = max(source.gsd_m, reference.gsd_m)
    ref_factor = gsd_work / reference.gsd_m
    src_factor = gsd_work / source.gsd_m

    if ref_factor > 1.001:
        ref = prefilter_for_downsample(ref, ref_factor)
        ref = cv2.resize(ref, (max(1, int(round(ref.shape[1] / ref_factor))),
                               max(1, int(round(ref.shape[0] / ref_factor)))),
                         interpolation=cv2.INTER_AREA)
    if src_factor > 1.001:
        src = prefilter_for_downsample(src, src_factor)
        src = cv2.resize(src, (max(1, int(round(src.shape[1] / src_factor))),
                               max(1, int(round(src.shape[0] / src_factor)))),
                         interpolation=cv2.INTER_AREA)

    # Working reference pixels are `ref_factor` times larger than the product's own, so
    # lifting a result back into the product frame is a pure scaling.
    frame_correction = np.diag([ref_factor, ref_factor, 1.0]).astype(np.float64)

    grid = OrthoGrid(
        crs=equirectangular(), transform=Affine(gsd_work, 0, 0, 0, -gsd_work, 0),
        height=ref.shape[0], width=ref.shape[1], gsd_m=gsd_work,
    )

    pad_h = max(0, ref.shape[0] - src.shape[0])
    pad_w = max(0, ref.shape[1] - src.shape[1])
    residual_px, shift = coarse_residual(np.pad(src, ((0, pad_h), (0, pad_w))), ref, 4)

    return PrealignResult(
        src_ortho=src, ref_ortho=ref,
        src_mask=np.ones(src.shape, bool), ref_mask=np.ones(ref.shape, bool),
        grid=grid,
        src_model=SensorModel("B", np.eye(3), grid.crs, source.pid, notes=["synthetic benchmark"]),
        ref_model=SensorModel("B", np.eye(3), grid.crs, reference.pid, notes=["synthetic benchmark"]),
        residual_px=residual_px, residual_shift=shift, accepted=True,
        frame_correction=frame_correction,
        meta={
            "skipped": bool(ref_factor <= 1.001 and src_factor <= 1.001),
            "reason": "synthetic benchmark pair",
            "gsd_work_m": round(gsd_work, 4),
            "reference_decimation": round(ref_factor, 4),
            "source_decimation": round(src_factor, 4),
        },
    )
