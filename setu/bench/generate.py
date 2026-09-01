"""S8 / Section 6.2 - the controlled illumination and scale benchmark. Novelty N5.

The problem statement's dataset is marked TBD, and even when real products arrive there
is no exact geometric truth for a cross-sensor pair. This module removes that
dependency: both images of a pair are rendered from the same terrain, and one is warped
by a homography that is *known exactly*. Registration error is therefore measurable as
true geometric error rather than as the residual of the model that was just fitted to
the data - which is the single easiest place for a project like this to be dishonest.

The two curves this produces answer the problem statement's two named challenges
directly: RMSE against delta sun-elevation, and RMSE against scale ratio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from itertools import product as iproduct
from pathlib import Path
from typing import Any, Iterator, Literal

import cv2
import numpy as np
from shapely.geometry import Polygon

from setu.bench.degrade import degrade, thermal_like
from setu.bench.terrain import TerrainPatch, get_terrain
from setu.illum.render import render_dem
from setu.types import IlluminationState, Product

WarpKind = Literal["similarity", "affine", "projective"]


@dataclass
class BenchPair:
    """One benchmark pair with exact ground truth.

    `H_true` maps *source pixel coordinates to reference pixel coordinates*, which is
    the same direction and the same frame the pipeline estimates, so a predicted
    transform can be compared with truth without any convention juggling.
    """

    pair_id: str
    source: np.ndarray
    reference: np.ndarray
    H_true: np.ndarray
    gsd_src_m: float
    gsd_ref_m: float
    illum_src: IlluminationState
    illum_ref: IlluminationState
    scale_ratio: float
    dem: np.ndarray | None = None
    dem_gsd_m: float | None = None
    #: The DEM crop co-registered with `reference`, pixel for pixel. Re-illumination needs
    #: *this*, not the full tile: rescaling the whole DEM onto the reference's shape would
    #: render a different piece of ground and quietly defeat the entire stage.
    dem_ref: np.ndarray | None = None
    terrain: str = "highland"
    cross_modal: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- truth API

    @property
    def d_sun_elev(self) -> float:
        return abs(self.illum_src.sun_elev_deg - self.illum_ref.sun_elev_deg)

    @property
    def d_sun_az(self) -> float:
        d = abs(self.illum_src.sun_az_deg - self.illum_ref.sun_az_deg) % 360.0
        return float(min(d, 360.0 - d))

    def map_true(self, pts_src: np.ndarray) -> np.ndarray:
        """Apply the true transform to source points, giving reference points."""
        return apply_h(self.H_true, pts_src)

    def truth_grid(self, n: int = 16, margin: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
        """A regular grid of exact correspondences, for checkpoint-style evaluation."""
        h, w = self.source.shape[:2]
        xs = np.linspace(margin * w, (1 - margin) * w, n)
        ys = np.linspace(margin * h, (1 - margin) * h, n)
        gx, gy = np.meshgrid(xs, ys)
        src = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)
        return src, self.map_true(src)

    def to_products(self) -> tuple[Product, Product]:
        """Wrap the pair as `Product`s so the normal pipeline can consume it."""
        return (
            _as_product(f"{self.pair_id}_src", self.source, self.gsd_src_m, self.illum_src),
            _as_product(f"{self.pair_id}_ref", self.reference, self.gsd_ref_m, self.illum_ref),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "terrain": self.terrain,
            "cross_modal": self.cross_modal,
            "scale_ratio": self.scale_ratio,
            "gsd_src_m": self.gsd_src_m,
            "gsd_ref_m": self.gsd_ref_m,
            "d_sun_elev_deg": round(self.d_sun_elev, 3),
            "d_sun_az_deg": round(self.d_sun_az, 3),
            "illum_src": self.illum_src.to_dict(),
            "illum_ref": self.illum_ref.to_dict(),
            "H_true": self.H_true.tolist(),
            "source_shape": list(self.source.shape),
            "reference_shape": list(self.reference.shape),
            **self.meta,
        }


def _as_product(pid: str, arr: np.ndarray, gsd: float, illum: IlluminationState) -> Product:
    h, w = arr.shape[:2]
    # A nominal equatorial footprint; the synthetic path never uses it for geolocation,
    # but downstream code is entitled to assume every Product carries one.
    deg = (max(h, w) * gsd) / 30_000.0
    poly = Polygon([(-deg, -deg), (deg, -deg), (deg, deg), (-deg, deg)])
    return Product(
        pid=pid, sensor="SYNTHETIC", array=arr.astype(np.float32), gsd_m=float(gsd),
        footprint=poly,
        corner_latlon=np.array([[deg, -deg], [deg, deg], [-deg, deg], [-deg, -deg]]),
        illum=illum, label={"synthetic": True},
    )


# ------------------------------------------------------------------- warping

def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an (N, 2) array of points."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.column_stack([p, np.ones(len(p))])
    out = hom @ np.asarray(H, dtype=np.float64).T
    w = np.where(np.abs(out[:, 2]) < 1e-12, 1e-12, out[:, 2])
    return np.column_stack([out[:, 0] / w, out[:, 1] / w])


def random_warp(
    kind: WarpKind,
    scale: float,
    centre: tuple[float, float],
    rng: np.random.Generator,
    max_rotation_deg: float = 12.0,
    max_shift_px: float = 24.0,
    shear: float = 0.03,
    projective: float = 2.5e-5,
) -> np.ndarray:
    """A random but bounded warp mapping source pixels to reference pixels.

    The scale factor is not random: it is the requested GSD ratio, so that the scale
    sweep measures exactly the quantity it claims to. Everything else - rotation, shift,
    shear and the projective terms - is sampled, so a method cannot succeed by having
    memorised one geometry.
    """
    cx, cy = centre
    theta = np.radians(rng.uniform(-max_rotation_deg, max_rotation_deg))
    tx, ty = rng.uniform(-max_shift_px, max_shift_px, 2)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    S = np.array([
        [scale * cos_t, -scale * sin_t, 0.0],
        [scale * sin_t,  scale * cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ])

    if kind in ("affine", "projective"):
        S = S @ np.array([[1.0, rng.uniform(-shear, shear), 0.0],
                          [rng.uniform(-shear, shear), 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
    if kind == "projective":
        S = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [rng.uniform(-projective, projective), rng.uniform(-projective, projective), 1.0]]) @ S

    # Rotate about the image centre rather than the origin, then add the shift.
    to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    back = np.array([[1.0, 0.0, cx * scale + tx], [0.0, 1.0, cy * scale + ty], [0.0, 0.0, 1.0]])
    H = back @ S @ to_origin
    return H / H[2, 2]


def warp_to_source_frame(scene: np.ndarray, H: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Resample a reference-frame scene into the source frame defined by H.

    `cv2.warpPerspective` with `WARP_INVERSE_MAP` evaluates dst(p) = scene(H p), which is
    precisely the definition of H as source-to-reference. Lanczos is used because area
    detail is being *removed* here, and a cheaper kernel would alias the crater rims into
    the very high-frequency structure the matchers key on.
    """
    return cv2.warpPerspective(
        np.asarray(scene, dtype=np.float32),
        np.asarray(H, dtype=np.float64),
        (out_shape[1], out_shape[0]),
        flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT101,
    )


# ------------------------------------------------------------ pair synthesis

def make_pair(
    patch: TerrainPatch,
    illum_src: IlluminationState,
    illum_ref: IlluminationState,
    scale_ratio: float = 1.0,
    warp_kind: WarpKind = "affine",
    src_sensor: str = "OHRC",
    ref_sensor: str = "NAC_L",
    cross_modal: bool = False,
    tile_px: int = 512,
    min_src_px: int = 160,
    pair_id: str = "pair",
    seed: int = 0,
    albedo: float = 0.12,
) -> BenchPair:
    """Render one benchmark pair with an exactly known transform.

    The reference is rendered at its own illumination on the terrain grid. The source is
    rendered at the source illumination, warped by H_true into a frame that is
    `scale_ratio` times coarser, and then degraded through the source payload's
    noise model. Never the other way round: upsampling the coarser image to match the
    finer one manufactures detail, and is exactly the failure mode MoonMetaSync documents.
    """
    rng = np.random.default_rng(seed)
    dem, gsd = patch.dem, patch.gsd_m

    ref_scene = render_dem(dem, gsd, illum_ref, albedo=albedo, rng=rng).image
    src_scene = render_dem(dem, gsd, illum_src, albedo=albedo, rng=rng).image

    # Both images cover the same ground; the coarser one simply has fewer pixels.
    # The reference crop is grown with the scale ratio so that the source keeps at least
    # `min_src_px` across, because a 40x ratio against a fixed 512 px reference would
    # leave a 13 px source and would be measuring nothing but interpolation.
    dem_edge = min(dem.shape[0], dem.shape[1])
    want_ref = int(round(max(tile_px, min_src_px * scale_ratio)))
    ref_h = ref_w = min(want_ref, dem_edge)
    r0 = (dem.shape[0] - ref_h) // 2
    c0 = (dem.shape[1] - ref_w) // 2
    reference = ref_scene[r0:r0 + ref_h, c0:c0 + ref_w]

    src_h = max(24, int(round(ref_h / scale_ratio)))
    src_w = max(24, int(round(ref_w / scale_ratio)))
    src_px_limited = src_h < min_src_px

    # H_true maps source pixels to *reference crop* pixels, which is the frame the
    # pipeline estimates in and the frame every metric is reported in.
    H_true = random_warp(warp_kind, scale_ratio, (src_w / 2.0, src_h / 2.0), rng)
    H_true /= H_true[2, 2]

    # Sampling happens against the uncropped scene, so the crop origin is composed in
    # for the warp only. Keeping these two transforms distinct is the difference between
    # a truth that is exact and one that is off by the crop offset.
    offset = np.array([[1.0, 0.0, c0], [0.0, 1.0, r0], [0.0, 0.0, 1.0]])
    H_sample = offset @ H_true

    source = warp_to_source_frame(src_scene, H_sample, (src_h, src_w))
    if cross_modal:
        dem_src = warp_to_source_frame(dem, H_sample, (src_h, src_w))
        source = thermal_like(source, dem_src, gsd * scale_ratio, rng)

    source = degrade(source, src_sensor, rng)
    reference = degrade(reference, ref_sensor, rng)

    return BenchPair(
        pair_id=pair_id,
        source=source,
        reference=reference,
        H_true=H_true,
        gsd_src_m=gsd * scale_ratio,
        gsd_ref_m=gsd,
        illum_src=illum_src,
        illum_ref=illum_ref,
        scale_ratio=float(scale_ratio),
        dem=dem,
        dem_gsd_m=gsd,
        dem_ref=np.ascontiguousarray(dem[r0:r0 + ref_h, c0:c0 + ref_w]),
        terrain=patch.source,
        cross_modal=cross_modal,
        meta={
            "warp_kind": warp_kind, "src_sensor": src_sensor, "ref_sensor": ref_sensor, "seed": seed,
            "source_px": [src_h, src_w], "reference_px": [ref_h, ref_w],
            # True when the DEM was too small to keep the source above min_src_px. The
            # pair is still valid and its truth is still exact, but the result belongs in
            # a footnote rather than in a headline average.
            "source_px_limited": bool(src_px_limited),
        },
    )


def sun_elevation_sweep(
    patch: TerrainPatch,
    ref_elev: float = 45.0,
    src_elevs: list[float] | None = None,
    sun_az: float = 135.0,
    az_offset: float = 0.0,
    scale_ratio: float = 1.0,
    seed: int = 26166,
    **kw: Any,
) -> Iterator[BenchPair]:
    """Sweep the source sun elevation against a fixed reference. Headline curve one.

    Only the sun geometry varies, so any change in RMSE or inlier ratio across the
    sweep is attributable to illumination and to nothing else.
    """
    for i, elev in enumerate(src_elevs or [10, 20, 30, 45, 60, 75]):
        yield make_pair(
            patch,
            illum_src=IlluminationState(sun_az_deg=sun_az + az_offset, sun_elev_deg=elev, source="synthetic"),
            illum_ref=IlluminationState(sun_az_deg=sun_az, sun_elev_deg=ref_elev, source="synthetic"),
            scale_ratio=scale_ratio,
            pair_id=f"sun_elev_{elev:g}",
            seed=seed + i,
            **kw,
        )


def scale_sweep(
    patch: TerrainPatch,
    ratios: list[float] | None = None,
    sun_az: float = 135.0,
    src_elev: float = 30.0,
    ref_elev: float = 45.0,
    seed: int = 26166,
    **kw: Any,
) -> Iterator[BenchPair]:
    """Sweep the GSD ratio at a fixed, moderate illumination difference. Headline curve two."""
    for i, s in enumerate(ratios or [1, 2, 4, 8, 16, 40, 100]):
        yield make_pair(
            patch,
            illum_src=IlluminationState(sun_az_deg=sun_az, sun_elev_deg=src_elev, source="synthetic"),
            illum_ref=IlluminationState(sun_az_deg=sun_az, sun_elev_deg=ref_elev, source="synthetic"),
            scale_ratio=float(s),
            pair_id=f"scale_{s:g}x",
            seed=seed + 100 + i,
            **kw,
        )


def full_grid(
    patch: TerrainPatch,
    sun_az: list[float],
    sun_elev: list[float],
    emission: list[float],
    scales: list[float],
    ref_elev: float = 45.0,
    ref_az: float = 135.0,
    seed: int = 26166,
    limit: int | None = None,
    **kw: Any,
) -> Iterator[BenchPair]:
    """The full (azimuth, elevation, emission, scale) sweep of Section 6.2."""
    n = 0
    for az, el, em, s in iproduct(sun_az, sun_elev, emission, scales):
        if limit is not None and n >= limit:
            return
        yield make_pair(
            patch,
            illum_src=IlluminationState(sun_az_deg=az, sun_elev_deg=el, emission_deg=em, source="synthetic"),
            illum_ref=IlluminationState(sun_az_deg=ref_az, sun_elev_deg=ref_elev, source="synthetic"),
            scale_ratio=float(s),
            pair_id=f"az{az:g}_el{el:g}_em{em:g}_s{s:g}",
            seed=seed + n,
            **kw,
        )
        n += 1


def write_bench(
    out_dir: str | Path,
    pairs: Iterator[BenchPair] | list[BenchPair],
    save_arrays: bool = True,
) -> dict[str, Any]:
    """Write a benchmark set to disk with a manifest, so runs are reproducible."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for pair in pairs:
        entry = pair.manifest()
        if save_arrays:
            npz = out / f"{pair.pair_id}.npz"
            np.savez_compressed(
                npz,
                source=pair.source,
                reference=pair.reference,
                H_true=pair.H_true,
                dem=pair.dem if pair.dem is not None else np.zeros((1, 1), np.float32),
                dem_ref=pair.dem_ref if pair.dem_ref is not None else np.zeros((1, 1), np.float32),
            )
            entry["file"] = npz.name
        manifest.append(entry)

    doc = {"n_pairs": len(manifest), "pairs": manifest}
    (out / "manifest.json").write_text(json.dumps(doc, indent=2))
    return doc


def load_bench_pair(npz_path: str | Path, manifest_entry: dict[str, Any]) -> BenchPair:
    """Reload a written pair, restoring its exact ground truth."""
    d = np.load(npz_path)
    ill = lambda k: IlluminationState(**{  # noqa: E731
        kk: vv for kk, vv in manifest_entry[k].items() if kk != "source"
    } | {"source": "synthetic"})
    dem = d["dem"]
    dem_ref = d["dem_ref"] if "dem_ref" in d else np.zeros((1, 1), np.float32)
    return BenchPair(
        pair_id=manifest_entry["pair_id"],
        source=d["source"], reference=d["reference"], H_true=d["H_true"],
        gsd_src_m=manifest_entry["gsd_src_m"], gsd_ref_m=manifest_entry["gsd_ref_m"],
        illum_src=ill("illum_src"), illum_ref=ill("illum_ref"),
        scale_ratio=manifest_entry["scale_ratio"],
        dem=dem if dem.size > 1 else None,
        dem_ref=dem_ref if dem_ref.size > 1 else None,
        terrain=manifest_entry.get("terrain", "unknown"),
        cross_modal=manifest_entry.get("cross_modal", False),
    )
