"""Lunar terrain for the controlled benchmark.

A real SLDEM2015 or NAC DTM tile is used whenever one is present. When it is not -
which is the normal case before the PRADAN and PDS downloads complete, and the case
the specification's risk table tells you to build against so that no phase is blocked
by data - this module synthesises a statistically lunar surface instead.

Synthetic here means the *shape* is generated, not the physics and not the metrics. The
renderer, the matcher, and every number in the evaluation harness operate identically
on a synthetic tile and on a real one, and the ground truth is exact either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class TerrainPatch:
    """One DEM patch plus the provenance the QA report has to state."""

    dem: np.ndarray
    gsd_m: float
    source: str
    lon_deg: float = 0.0
    lat_deg: float = 0.0
    meta: dict | None = None


def fractal_surface(
    size: int,
    gsd_m: float,
    beta: float = 4.2,
    rms_slope_deg: float = 12.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Self-affine background topography with a 1/f^beta power spectrum.

    Parameterised by RMS slope rather than RMS height, because slope is what the
    photometry actually sees and it is the statistic that stays meaningful when the
    patch size or the GSD changes. Height is solved for afterwards by rescaling to hit
    the requested slope, so a 512 px tile and a 2048 px tile of the same terrain class
    are directly comparable.

    `beta` near 4 corresponds to a Hurst exponent around 1, the upper end of the range
    measured on real lunar surfaces, and gives a field whose roughness is set by the
    crater population rather than by numerical high-frequency noise.
    """
    rng = rng or np.random.default_rng(0)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    f = np.sqrt(fy**2 + fx**2)
    f[0, 0] = 1.0

    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, (size, size))
    field = np.fft.ifft2(amp * np.exp(1j * phase)).real
    field -= field.mean()

    gy, gx = np.gradient(field, gsd_m, gsd_m)
    current = float(np.sqrt(np.mean(gx**2 + gy**2)))
    target = float(np.tan(np.radians(rms_slope_deg)))
    scale = target / current if current > 1e-12 else 0.0
    return (field * scale).astype(np.float32)


def crater_profile(r_norm: np.ndarray, depth_m: float, rim_height_m: float) -> np.ndarray:
    """Radial profile of a simple bowl crater with a raised rim and ejecta skirt.

    Inside the rim the floor follows the parabolic bowl that fresh simple craters
    actually show; outside it the ejecta blanket decays as roughly r^-3, which is the
    standard empirical fall-off.
    """
    z = np.zeros_like(r_norm, dtype=np.float32)

    inside = r_norm <= 1.0
    z[inside] = -depth_m * (1.0 - r_norm[inside] ** 2) + rim_height_m * r_norm[inside] ** 4

    outside = (r_norm > 1.0) & (r_norm < 3.0)
    z[outside] = rim_height_m * np.power(r_norm[outside], -3.0)
    return z


def crater_population(
    size: int,
    gsd_m: float,
    density: float = 1.4e-4,
    d_min_px: float = 4.0,
    d_max_frac: float = 0.35,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Stamp a crater population following an equilibrium size-frequency distribution.

    Counts follow N(>D) proportional to D^-2, the saturation slope for small lunar
    craters, so the field is dominated by many small craters with a few large ones -
    which is exactly the self-similar, repetitive texture that makes deep matchers
    produce confident wrong matches, and therefore what the benchmark has to contain.
    """
    rng = rng or np.random.default_rng(0)
    dem = np.zeros((size, size), dtype=np.float32)
    d_max = d_max_frac * size

    n_craters = max(8, int(density * size * size))
    u = rng.uniform(size=n_craters)
    # Inverse transform of a D^-2 differential distribution between d_min and d_max.
    diameters = d_min_px * d_max / (d_max - u * (d_max - d_min_px))
    diameters = np.clip(diameters, d_min_px, d_max)

    catalogue: list[dict] = []
    yy, xx = np.mgrid[0:size, 0:size]

    for d_px in np.sort(diameters)[::-1]:
        cy = rng.uniform(0, size)
        cx = rng.uniform(0, size)
        radius = d_px / 2.0
        # Depth-to-diameter near 0.2 for fresh simple craters, degraded by a random
        # freshness factor so the field mixes crisp and subdued morphologies.
        freshness = rng.uniform(0.25, 1.0)
        depth = 0.2 * d_px * gsd_m * freshness
        rim = 0.04 * d_px * gsd_m * freshness

        r0, r1 = int(max(0, cy - 3 * radius)), int(min(size, cy + 3 * radius + 1))
        c0, c1 = int(max(0, cx - 3 * radius)), int(min(size, cx + 3 * radius + 1))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        rn = np.hypot(yy[r0:r1, c0:c1] - cy, xx[r0:r1, c0:c1] - cx) / max(radius, 1e-6)
        dem[r0:r1, c0:c1] += crater_profile(rn, depth, rim)

        catalogue.append({
            "cx_px": float(cx), "cy_px": float(cy),
            "diameter_px": float(d_px), "diameter_m": float(d_px * gsd_m),
            "depth_m": float(depth), "freshness": float(freshness),
        })

    return dem, catalogue


def synthetic_terrain(
    size: int = 1024,
    gsd_m: float = 5.0,
    terrain: str = "highland",
    seed: int = 26166,
) -> TerrainPatch:
    """Build a synthetic lunar DEM patch.

    `highland` is heavily cratered with strong regional relief; `mare` is smoother with
    a sparser crater population; `selfsimilar` is a deliberately repetitive field used
    to stress the agreement gate, per the risk table.
    """
    rng = np.random.default_rng(seed)
    # RMS slopes here are the values reported for the corresponding lunar terrain
    # classes from LOLA and NAC DTM analyses at comparable baselines.
    params = {
        "highland": dict(beta=4.2, slope=11.0, density=2.2e-4),
        "mare": dict(beta=4.0, slope=3.5, density=6.0e-5),
        "selfsimilar": dict(beta=4.6, slope=5.0, density=4.0e-4),
    }[terrain]

    base = fractal_surface(size, gsd_m, beta=params["beta"], rms_slope_deg=params["slope"], rng=rng)
    craters, catalogue = crater_population(size, gsd_m, density=params["density"], rng=rng)
    dem = gaussian_filter(base + craters, 0.7).astype(np.float32)

    return TerrainPatch(
        dem=dem,
        gsd_m=gsd_m,
        source=f"synthetic:{terrain}",
        meta={
            "terrain": terrain, "seed": seed, "size_px": size,
            "n_craters": len(catalogue),
            "relief_m": float(dem.max() - dem.min()),
            "rms_slope_deg": rms_slope_deg(dem, gsd_m),
            "median_slope_deg": median_slope_deg(dem, gsd_m),
        },
    )


def load_dem(path: str | Path, max_size: int = 2048) -> TerrainPatch:
    """Load a real DEM tile (SLDEM2015, NAC DTM, Kaguya TC DEM) via rasterio."""
    import rasterio
    from rasterio.enums import Resampling

    path = Path(path)
    with rasterio.open(path) as ds:
        scale = max(1, int(max(ds.height, ds.width) / max_size))
        out_h, out_w = ds.height // scale, ds.width // scale
        dem = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.average).astype(np.float32)
        gsd = float(abs(ds.transform.a)) * scale
        cx, cy = ds.transform * (ds.width / 2, ds.height / 2)
        nodata = ds.nodata

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
        if np.isnan(dem).any():
            dem = np.where(np.isnan(dem), float(np.nanmedian(dem)), dem)

    return TerrainPatch(
        dem=dem, gsd_m=gsd, source=f"dem:{path.name}",
        lon_deg=float(cx), lat_deg=float(cy),
        meta={"path": str(path), "downsample": scale, "relief_m": float(dem.max() - dem.min())},
    )


def get_terrain(
    dem_path: str | Path | None = None,
    size: int = 1024,
    gsd_m: float = 5.0,
    terrain: str = "highland",
    seed: int = 26166,
) -> TerrainPatch:
    """Real DEM when one is supplied and readable, synthetic otherwise.

    A DEM that fails to open falls through to synthesis with the reason recorded, so a
    missing tile degrades the provenance string rather than the run.
    """
    if dem_path is not None and Path(dem_path).exists():
        try:
            return load_dem(dem_path)
        except Exception as exc:
            patch = synthetic_terrain(size, gsd_m, terrain, seed)
            patch.meta = {**(patch.meta or {}), "dem_load_failed": f"{type(exc).__name__}: {exc}"}
            return patch
    return synthetic_terrain(size, gsd_m, terrain, seed)


def rms_slope_deg(dem: np.ndarray, gsd_m: float) -> float:
    """RMS surface slope in degrees, the standard lunar roughness statistic."""
    gy, gx = np.gradient(np.asarray(dem, dtype=np.float64), gsd_m, gsd_m)
    return float(np.degrees(np.arctan(np.sqrt(np.mean(gx**2 + gy**2)))))


def median_slope_deg(dem: np.ndarray, gsd_m: float) -> float:
    gy, gx = np.gradient(np.asarray(dem, dtype=np.float64), gsd_m, gsd_m)
    return float(np.degrees(np.arctan(np.median(np.hypot(gx, gy)))))
