"""S2a - Sun-synchronised reference re-illumination. Novelty N1.

This is the component the whole project turns on. Rather than hunting for a descriptor
that survives a sun-angle change, SETU removes the sun-angle change: the reference is
re-rendered from the terrain model under the exact solar azimuth, solar elevation and
emission angle recorded in the Chandrayaan-2 metadata, so that matching happens between
two images that already agree on illumination.

What the render is and is not. At 59 m SLDEM2015 resolution against 0.25 m OHRC pixels,
the rendering is not photorealistic and is not claimed to be. It carries structure at
the scale the DEM supports, which is what constrains the global and mid-frequency
alignment; sub-pixel accuracy comes from S4 operating on the real image pair. Where a
NAC DTM exists the render is correspondingly sharper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from setu.illum.reflectance import (
    emission_from_normals,
    incidence_from_normals,
    lunar_lambert,
    view_vector,
)
from setu.illum.shadow import shadow_mask, surface_normals
from setu.types import IlluminationState


@dataclass
class RenderResult:
    """A rendered reference plus the diagnostics the QA report needs."""

    image: np.ndarray
    shadow: np.ndarray
    incidence_cos: np.ndarray
    emission_cos: np.ndarray
    illum: IlluminationState
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shadow_fraction(self) -> float:
        return float(self.shadow.mean())


def dem_to_mesh(dem: np.ndarray, gsd_m: float):
    """Triangle mesh from a DEM patch, pixel centres as vertices.

    Only the optional Embree shadow path needs this; the default render works
    directly on the grid.
    """
    import trimesh

    h, w = dem.shape
    yy, xx = np.mgrid[0:h, 0:w]
    verts = np.column_stack([
        (xx.ravel() * gsd_m).astype(np.float64),
        ((h - 1 - yy).ravel() * gsd_m).astype(np.float64),
        np.asarray(dem, dtype=np.float64).ravel(),
    ])
    idx = np.arange(h * w).reshape(h, w)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def render_dem(
    dem: np.ndarray,
    gsd_m: float,
    illum: IlluminationState,
    albedo: float | np.ndarray = 0.12,
    shadow_method: str = "horizon",
    psf_sigma_px: float = 0.5,
    noise_sigma: float = 0.0,
    view_azimuth_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> RenderResult:
    """Render a DEM patch under one illumination state.

    Steps follow S2a: normals, collimated sun vector, Lunar-Lambert reflectance,
    ray-cast shadows, sensor PSF, and finally noise matched to the source image so the
    two sides of the match have comparable texture statistics rather than one being
    suspiciously clean.
    """
    dem = np.asarray(dem, dtype=np.float32)
    rng = rng or np.random.default_rng(0)

    normals = surface_normals(dem, gsd_m)
    sun = illum.sun_vector()
    mu0 = np.clip(incidence_from_normals(normals, sun), 0.0, 1.0)

    view = view_vector(illum.emission_deg, view_azimuth_deg)
    mu = np.clip(emission_from_normals(normals, view), 1e-3, 1.0)

    shadow = shadow_mask(dem, gsd_m, illum.sun_az_deg, illum.sun_elev_deg, method=shadow_method)

    radiance = lunar_lambert(mu0, mu, illum.phase_deg or 0.0, albedo=albedo)
    # A shadowed facet still receives a little scattered light from surrounding slopes.
    # Zeroing it outright produces hard black holes that no real image contains and that
    # then dominate the structural transforms in S2b.
    radiance = np.where(shadow, radiance * 0.04, radiance).astype(np.float32)

    if psf_sigma_px > 0:
        radiance = gaussian_filter(radiance, psf_sigma_px).astype(np.float32)
    if noise_sigma > 0:
        radiance = (radiance + rng.normal(0.0, noise_sigma, radiance.shape)).astype(np.float32)

    return RenderResult(
        image=radiance,
        shadow=shadow,
        incidence_cos=mu0.astype(np.float32),
        emission_cos=mu.astype(np.float32),
        illum=illum,
        meta={
            "gsd_m": float(gsd_m),
            "sun_az_deg": illum.sun_az_deg,
            "sun_elev_deg": illum.sun_elev_deg,
            "emission_deg": illum.emission_deg,
            "phase_deg": illum.phase_deg,
            "albedo": float(np.mean(albedo)),
            "shadow_method": shadow_method,
            "shadow_fraction": float(shadow.mean()),
            "psf_sigma_px": float(psf_sigma_px),
            "noise_sigma": float(noise_sigma),
        },
    )


def reilluminate_reference(
    dem: np.ndarray,
    gsd_m: float,
    source_illum: IlluminationState,
    reference_image: np.ndarray | None = None,
    albedo_map: np.ndarray | None = None,
    shadow_method: str = "horizon",
    psf_sigma_px: float = 0.5,
    match_noise: bool = True,
    source_image: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> RenderResult:
    """Render the reference at the *source* image's illumination geometry.

    When a real reference image is supplied its low-frequency content is used as the
    albedo map, so genuine bright and dark terrain survives into the render instead of
    being flattened to a single constant. That matters on mare/highland boundaries,
    where a constant albedo would invent an edge that is not in the source image.
    """
    dem = np.asarray(dem, dtype=np.float32)

    if albedo_map is None and reference_image is not None:
        albedo_map = estimate_albedo(reference_image)
    albedo: float | np.ndarray = 0.12 if albedo_map is None else albedo_map

    noise_sigma = 0.0
    if match_noise and source_image is not None:
        noise_sigma = estimate_noise_sigma(source_image) * float(np.mean(albedo))

    return render_dem(
        dem,
        gsd_m,
        source_illum,
        albedo=albedo,
        shadow_method=shadow_method,
        psf_sigma_px=psf_sigma_px,
        noise_sigma=noise_sigma,
        rng=rng,
    )


def estimate_albedo(reference_image: np.ndarray, sigma_px: float = 24.0, base: float = 0.12) -> np.ndarray:
    """Low-frequency albedo proxy from a reference image.

    Heavy blurring is deliberate. Anything sharper than the DEM can resolve is shading,
    not albedo, and feeding shading back in as albedo would bake the reference's own sun
    angle into the render - defeating the entire point of re-illuminating it.
    """
    img = np.asarray(reference_image, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    low = gaussian_filter(img, sigma_px)
    med = float(np.median(low)) or 1.0
    return np.clip(base * low / med, 0.02, 0.4).astype(np.float32)


def estimate_noise_sigma(image: np.ndarray) -> float:
    """Robust noise estimate via the median absolute deviation of a Laplacian.

    The 0.6745 factor converts MAD to a Gaussian sigma; the 1/sqrt(6) accounts for the
    variance amplification of the 4-neighbour Laplacian kernel.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    lap = (
        -4.0 * img
        + np.roll(img, 1, 0) + np.roll(img, -1, 0)
        + np.roll(img, 1, 1) + np.roll(img, -1, 1)
    )[1:-1, 1:-1]
    mad = float(np.median(np.abs(lap - np.median(lap))))
    return float(mad / 0.6745 / np.sqrt(6.0))


def render_similarity(rendered: np.ndarray, real: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """How well a render matches a real image at the same geometry.

    The P5 gate is NCC > 0.6 between the rendered reference and a real image acquired
    at the same illumination, which is the check that the photometry is right rather
    than merely plausible.
    """
    a = np.asarray(rendered, dtype=np.float64)
    b = np.asarray(real, dtype=np.float64)
    if a.shape != b.shape:
        # Purely a diagnostic, so a scale difference is resampled away rather than
        # raising. The coarser grid wins, because upsampling to compare would invent
        # detail and inflate the very correlation being reported.
        import cv2

        if mask is not None and mask.shape != a.shape:
            mask = None
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
        b = cv2.resize(b.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
        if mask is not None:
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if mask is not None:
        a, b = a[mask], b[mask]
    else:
        a, b = a.ravel(), b.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 16:
        return {"ncc": float("nan"), "n": int(a.size)}
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    ncc = float((a * b).sum() / denom) if denom > 0 else float("nan")
    return {"ncc": ncc, "n": int(a.size)}
