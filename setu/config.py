"""Resolved configuration for one SETU run.

The specification is explicit that no magic numbers may live in code: every
threshold quoted in Sections 5 and 7 appears here as a named, documented field, one
YAML file supplies the per-experiment overrides, and the resolved model is written
into the run directory so a result can always be traced back to the settings that
produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PrealignConfig(BaseModel):
    """S1 - geometric pre-alignment."""

    gsd_scale_k: float = Field(1.0, description="gsd_work = max(gsd_src, gsd_ref) * k")
    polar_lat_deg: float = Field(60.0, description="Above this |latitude| use polar stereographic")
    oblique_extent_km: float = Field(50.0, description="Above this footprint extent use oblique stereographic")
    apply_parallax: bool = Field(True, description="Tier B terrain-parallax correction")
    sensor_model_tier: Literal["A", "B", "auto"] = "auto"
    #: S1 acceptance test - residual after pre-alignment must be a pure translation
    #: no larger than this, checked by coarse phase correlation on a 4x downsample.
    max_residual_px: float = 500.0
    acceptance_downsample: int = 4
    raise_on_acceptance_failure: bool = True
    max_size_px: int = Field(4096, description="Cap on the working ortho grid edge")


class IllumConfig(BaseModel):
    """S2 - illumination harmonisation."""

    reilluminate: bool = True
    albedo: float = Field(0.12, description="Constant lunar albedo when no map is supplied")
    shadow_method: Literal["horizon", "embree", "none"] = "horizon"
    psf_sigma_px: float = 0.5
    match_noise: bool = Field(True, description="Add source-matched noise to the render")
    structural: Literal["pc", "mim", "cfog", "lnift", "none"] = "pc"
    pc_scales: int = 4
    pc_orientations: int = 6
    pc_min_wavelength: float = 3.0
    pc_mult: float = 2.1
    pc_sigma_onf: float = 0.55
    pc_k: float = Field(2.0, description="Kovesi noise-rejection k; raised on low-SNR data")
    pc_noise_adaptive: bool = True
    lnift_window: int = 31
    cfog_orientations: int = 8
    cfog_sigma: float = 2.0
    iirs_band_lo_nm: float = 900.0
    iirs_band_hi_nm: float = 1600.0
    iirs_destripe: bool = True


class MatchConfig(BaseModel):
    """S3 - correspondence, two tracks plus the agreement gate."""

    track_a: bool = True
    track_b: bool = True
    #: Which representation each track consumes. Track A is a pretrained network and
    #: performs best on photometrically harmonised imagery that still looks natural; a
    #: six-level index map is far outside its training distribution. Track B is a
    #: handcrafted structural matcher and is built for the structural maps. Both are
    #: exposed because which one wins is a measurement, not an assumption.
    track_a_input: Literal["photometric", "structural"] = "photometric"
    track_b_input: Literal["photometric", "structural"] = "structural"
    deep_matcher: Literal["matchanything_eloftr", "matchanything_roma", "loftr", "lightglue", "xfeat", "disk", "aliked", "auto"] = "auto"
    deep_weights_dir: str | None = None
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    tile_size: int = 640
    tile_overlap: float = 0.5
    deep_conf_threshold: float = 0.2
    max_matches_per_tile: int = 400
    estimate_rotation: bool = True
    # Track B
    lattice_detect: tuple[int, int] = (12, 12)
    detect_per_cell: int = 12
    mim_patch: int = 96
    lowe_ratio: float = Field(0.85, description="Looser than 0.75; cross-modal descriptors are less discriminative")
    template_window_px: int = 48
    template_metric: Literal["cfog", "ncc"] = "cfog"
    # Agreement gate
    agreement_tau_px: float = Field(2.0, description="Both tracks must land within tau of each other")
    sharpness_peak_ratio: float = Field(1.5, description="peak / second_peak for a single-track accept")
    sharpness_peak_min: float = Field(0.4, description="Absolute peak floor for a single-track accept")


class RefineConfig(BaseModel):
    """S4 - sub-pixel refinement and per-point covariance."""

    #: Which representation sub-pixel refinement runs on. The specification calls for the
    #: structural maps, because they are illumination-stable; that holds for phase
    #: congruency, which is continuous. The maximum index map is *quantised* - it takes
    #: six discrete values - and correlating a piecewise-constant image cannot resolve a
    #: sub-pixel shift. So the choice is exposed and measured rather than fixed, and the
    #: default structural transform is the continuous one.
    input: Literal["structural", "photometric"] = "structural"
    patch: int = 64
    upsample_factor: int = 50
    lsm: bool = True
    lsm_max_iter: int = 12
    lsm_tol_px: float = 1e-3
    covariance: Literal["lsm", "paraboloid", "auto"] = "auto"
    max_shift_px: float = Field(4.0, description="Reject a refinement that moves a point further than this")
    max_sigma_px: float = Field(0.5, description="Reject points whose sqrt(trace Sigma) exceeds this")


class ModelConfig(BaseModel):
    """S5 - outlier rejection and transformation model."""

    global_model: Literal["affine", "homography", "similarity", "auto"] = "auto"
    homography_emission_deg: float = Field(10.0, description="Switch to projective above this emission difference")
    similarity_min_inliers: int = Field(30, description="Fall back to similarity below this inlier count")
    ransac_confidence: float = 0.9999
    ransac_max_iters: int = 10000
    adaptive_threshold_scale: float = Field(3.0, description="thr = scale * median(sqrt(trace Sigma))")
    threshold_floor_px: float = 0.75
    threshold_ceiling_px: float = 8.0
    local_model: Literal["tps", "poly2", "none"] = "tps"
    tps_lambda_grid: list[float] = Field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    jitter_model: bool = True
    jitter_spline_knots: int = 12
    feedback_s5_to_s1: bool = True
    feedback_max_iters: int = Field(1, description="One iteration is enough; two is the cap")


class UniformConfig(BaseModel):
    """S6 - uniformity as an explicit objective."""

    lattice: tuple[int, int] = (8, 8)
    auto_lattice: bool = Field(True, description="cells per side ~ sqrt(target_points / 4)")
    target_points: int = 400
    per_cell_quota: int = 8
    anms_radius_px: float = 24.0
    reseed: bool = True
    reseed_passes: int = 2
    reseed_window_px: int = 5
    reseed_threshold_scale: float = 0.6
    coverage_target: float = 0.90


class ProductConfig(BaseModel):
    """S7 - output products."""

    resample: Literal["lanczos", "cubic", "bilinear"] = "lanczos"
    preview_resample: Literal["cubic", "bilinear"] = "cubic"
    write_geotiff: bool = True
    write_cog: bool = True
    write_geojson: bool = True
    write_pds4_label: bool = True
    write_report: bool = True
    write_blink_gif: bool = True
    blink_frames: int = 8
    preview_max_px: int = 1400


class EvalConfig(BaseModel):
    """S8 / Section 7 - evaluation protocol."""

    bootstrap_n: int = 1000
    bootstrap_alpha: float = 0.05
    success_thresholds_px: list[float] = Field(default_factory=lambda: [1.0, 3.0, 5.0, 10.0])
    ce_percentile: float = 90.0
    loocv: bool = True
    checkpoint_file: str | None = None
    baselines: list[str] = Field(
        default_factory=lambda: [
            "sift", "orb", "intfeat", "rift", "cfog",
            "superpoint_lightglue", "loftr", "matchanything",
            "setu_no_reillum", "setu_no_gate", "setu_no_uniform", "setu_full",
        ]
    )


class BenchConfig(BaseModel):
    """Section 6.2 - the controlled synthetic benchmark."""

    sun_az_deg: list[float] = Field(default_factory=lambda: [0, 45, 90, 135, 180, 225, 270, 315])
    sun_elev_deg: list[float] = Field(default_factory=lambda: [10, 20, 30, 45, 60, 75])
    emission_deg: list[float] = Field(default_factory=lambda: [0, 10, 25])
    scale_ratios: list[float] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 40, 100])
    tile_px: int = 512
    dem_px: int = 1024
    seed: int = 26166


class SetuConfig(BaseSettings):
    """The one resolved configuration object."""

    model_config = SettingsConfigDict(env_prefix="SETU_", extra="ignore")

    name: str = "default"
    description: str = ""
    seed: int = 26166
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"

    prealign: PrealignConfig = Field(default_factory=PrealignConfig)
    illum: IllumConfig = Field(default_factory=IllumConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)
    refine: RefineConfig = Field(default_factory=RefineConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    uniform: UniformConfig = Field(default_factory=UniformConfig)
    product: ProductConfig = Field(default_factory=ProductConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    bench: BenchConfig = Field(default_factory=BenchConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> SetuConfig:
        """Load `configs/default.yaml`, then the named experiment file, then overrides."""
        data: dict[str, Any] = {}
        default = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        if default.exists():
            data = _deep_merge(data, yaml.safe_load(default.read_text()) or {})
        if path is not None:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"config not found: {p}")
            data = _deep_merge(data, yaml.safe_load(p.read_text()) or {})
        data = _deep_merge(data, overrides)
        return cls(**data)

    def resolved(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.resolved(), sort_keys=False))


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
