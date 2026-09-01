"""The SETU pipeline: S0 through S7, with both feedback edges.

    S0 ingest -> S1 pre-align -> S2 illumination harmonisation -> S3 correspondence
       -> S4 sub-pixel refinement -> S5 outlier rejection and model
       -> S6 uniformity enforcement -> S7 products

    S5 -> S1 : once a global transform exists, pre-alignment is re-run from the
               corrected footprint. One iteration; two is the cap.
    S6 -> S3 : cells that fail their quota are re-seeded with a lowered acceptance
               threshold and a smaller search window, because the global model is now
               known. This is what actually delivers uniformity.

Every stage appends a record to `RunResult.stages`, so the QA report and the web demo
both narrate the run from the same source rather than from a re-derivation of it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from setu.config import SetuConfig
from setu.geom.prealign import PrealignResult, prealign, prealign_synthetic
from setu.illum.render import RenderResult, estimate_noise_sigma, reilluminate_reference, render_similarity
from setu.illum.structural import structural_transform
from setu.io.registry import check_pairing
from setu.match.base import MatchSet
from setu.match.deep import build_matcher, track_a_status
from setu.match.gate import GateResult, agreement_gate
from setu.match.structural import StructuralMatcher
from setu.match.tiling import iter_tiles
from setu.model.jitter import fit_jitter
from setu.model.local import fit_local
from setu.model.robust import fit_global_auto, mark_inliers
from setu.refine.refine import refine_matches
from setu.types import Product, RunResult, TiePoint
from setu.uniform.lattice import auto_lattice_size, build_lattice
from setu.uniform.reseed import apply_quota, reseed_empty_cells
from setu.uniform.stats import uniformity_report

ProgressFn = Callable[[str, str, dict[str, Any]], None]


@dataclass
class StageRecord:
    """One stage's outcome, timing and headline numbers."""

    stage: str
    label: str
    seconds: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "label": self.label, "seconds": round(self.seconds, 3), **self.detail}


class Pipeline:
    """One configured SETU run."""

    def __init__(self, config: SetuConfig, progress: ProgressFn | None = None) -> None:
        self.cfg = config
        self.progress = progress or (lambda stage, label, detail: None)
        self.stages: list[StageRecord] = []
        self._t0 = 0.0

    # ------------------------------------------------------------- plumbing

    def _begin(self, stage: str, label: str) -> None:
        self._t0 = time.perf_counter()
        self.progress(stage, label, {"status": "running"})

    def _end(self, stage: str, label: str, **detail: Any) -> StageRecord:
        rec = StageRecord(stage, label, time.perf_counter() - self._t0, detail)
        self.stages.append(rec)
        self.progress(stage, label, {"status": "done", **rec.to_dict()})
        return rec

    # ------------------------------------------------------------------ run

    def run(
        self,
        source: Product,
        reference: Product,
        dem: np.ndarray | None = None,
        dem_gsd_m: float | None = None,
        run_id: str | None = None,
        synthetic: bool = False,
    ) -> RunResult:
        """Register `source` against `reference` and return everything the run produced."""
        started = time.perf_counter()
        run_id = run_id or f"{source.pid}__{reference.pid}__{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

        # ---------------------------------------------------------- S0 ingest
        self._begin("S0", "Ingest")
        pairing = check_pairing(source, reference)
        self._end("S0", "Ingest",
                  source=source.summary(), reference=reference.summary(), pairing=pairing)

        # ------------------------------------------------------ S1 pre-align
        pre = self._stage_prealign(source, reference, dem, dem_gsd_m, synthetic, correction=None)

        # ------------------------------------------- S2 illumination + S3..S6
        result = self._register(source, reference, pre, dem, dem_gsd_m)

        # ---------------------------------------------- S5 -> S1 feedback edge
        if (
            self.cfg.model.feedback_s5_to_s1
            and not synthetic
            and result["global_fit"] is not None
            and result["global_fit"].n_inliers >= 12
        ):
            for _ in range(self.cfg.model.feedback_max_iters):
                self._begin("S5>S1", "Re-project from the corrected footprint")
                correction = _pixel_to_map_correction(result["global_fit"].H, pre.grid)
                try:
                    pre2 = prealign(
                        source, reference, dem, dem_gsd_m,
                        gsd_scale_k=self.cfg.prealign.gsd_scale_k,
                        polar_lat_deg=self.cfg.prealign.polar_lat_deg,
                        apply_parallax=self.cfg.prealign.apply_parallax,
                        sensor_model_tier=self.cfg.prealign.sensor_model_tier,
                        max_size_px=self.cfg.prealign.max_size_px,
                        max_residual_px=self.cfg.prealign.max_residual_px,
                        raise_on_failure=False,
                        correction=correction,
                    )
                except Exception as exc:
                    self._end("S5>S1", "Re-project from the corrected footprint",
                              applied=False, reason=f"{type(exc).__name__}: {exc}")
                    break

                improved = pre2.residual_px < pre.residual_px
                self._end("S5>S1", "Re-project from the corrected footprint",
                          applied=improved,
                          residual_before_px=round(pre.residual_px, 3),
                          residual_after_px=round(pre2.residual_px, 3))
                if not improved:
                    break
                pre = pre2
                result = self._register(source, reference, pre, dem, dem_gsd_m)

        # ------------------------------------------------------------ output
        # Anything SETU reports is expressed in the reference product's own pixel frame,
        # not in whatever working grid the run happened to use.
        correction = np.asarray(pre.frame_correction, dtype=np.float64)
        if not np.allclose(correction, np.eye(3)):
            _lift_to_product_frame(result, correction)

        metrics = self._metrics(result, pre, source, reference, time.perf_counter() - started)
        transform = {
            "global": result["global_fit"].to_dict() if result["global_fit"] else None,
            "local": result["local_model"].to_dict() if result["local_model"] else None,
            "jitter": result["jitter"].to_dict() if result["jitter"] else None,
            "prealign": pre.to_dict(),
            "grid": pre.grid.to_dict(),
        }

        return RunResult(
            run_id=run_id,
            source=source.summary(),
            reference=reference.summary(),
            tiepoints=result["tiepoints"],
            metrics=metrics,
            transform=transform,
            config=self.cfg.resolved(),
            stages=[s.to_dict() for s in self.stages],
        )

    # ------------------------------------------------------------ S1 wrapper

    def _stage_prealign(self, source, reference, dem, dem_gsd_m, synthetic, correction) -> PrealignResult:
        self._begin("S1", "Geometric pre-alignment")
        if synthetic:
            pre = prealign_synthetic(source, reference)
        else:
            pre = prealign(
                source, reference, dem, dem_gsd_m,
                gsd_scale_k=self.cfg.prealign.gsd_scale_k,
                polar_lat_deg=self.cfg.prealign.polar_lat_deg,
                oblique_extent_km=self.cfg.prealign.oblique_extent_km,
                apply_parallax=self.cfg.prealign.apply_parallax,
                sensor_model_tier=self.cfg.prealign.sensor_model_tier,
                max_size_px=self.cfg.prealign.max_size_px,
                max_residual_px=self.cfg.prealign.max_residual_px,
                acceptance_downsample=self.cfg.prealign.acceptance_downsample,
                raise_on_failure=self.cfg.prealign.raise_on_acceptance_failure,
                correction=correction,
            )
        self._end("S1", "Geometric pre-alignment", **pre.to_dict())
        return pre

    # ---------------------------------------------------------- S2 .. S6 core

    def _register(self, source, reference, pre, dem, dem_gsd_m) -> dict[str, Any]:
        src_img, ref_img = pre.src_ortho, pre.ref_ortho

        # ------------------------------------------------------------ S2 (a)
        self._begin("S2a", "Sun-synchronised re-illumination")
        render: RenderResult | None = None
        ref_for_matching = ref_img
        if self.cfg.illum.reilluminate and dem is not None:
            render = _render_at_native_resolution(
                dem, dem_gsd_m, pre.grid.gsd_m, ref_img, src_img, source.illum, self.cfg,
            )
            ref_for_matching = render.image
            sim_before = render_similarity(ref_img, src_img)["ncc"]
            sim_after = render_similarity(render.image, src_img)["ncc"]
            self._end("S2a", "Sun-synchronised re-illumination",
                      reillumination=True,
                      ncc_source_vs_real_reference=round(float(sim_before), 4),
                      ncc_source_vs_rendered_reference=round(float(sim_after), 4),
                      render=render.meta)
        else:
            self._end("S2a", "Sun-synchronised re-illumination",
                      reillumination=False,
                      reason="no terrain model supplied for this pair"
                      if dem is None else "disabled in configuration")

        # ------------------------------------------------------------ S2 (b)
        self._begin("S2b", "Structural transform")
        kind = self.cfg.illum.structural
        src_repr = structural_transform(src_img, kind)
        ref_repr = structural_transform(ref_for_matching, kind)
        self._end("S2b", "Structural transform",
                  transform=kind,
                  ncc_raw=round(float(render_similarity(src_img, ref_for_matching)["ncc"]), 4),
                  ncc_structural=round(float(render_similarity(src_repr, ref_repr)["ncc"]), 4),
                  source_noise_sigma=round(estimate_noise_sigma(src_img), 6))

        # Each track is handed the representation it was designed for. Track A is a
        # pretrained network: after S2a the two images already agree on illumination, and
        # feeding it a quantised index map instead costs accuracy for nothing. Track B and
        # the sub-pixel refinement of S4 run on the structural maps, which is where their
        # invariance comes from.
        inputs = {
            "photometric": (src_img, ref_for_matching),
            "structural": (src_repr, ref_repr),
        }
        a_src, a_ref = inputs[self.cfg.match.track_a_input]
        b_src, b_ref = inputs[self.cfg.match.track_b_input]

        # -------------------------------------------------------------- S3 A
        self._begin("S3A", "Track A: dense deep matcher")
        track_a = self._run_track_a(a_src, a_ref, pre)
        self._end("S3A", "Track A: dense deep matcher",
                  n_matches=len(track_a), input=self.cfg.match.track_a_input,
                  **{k: v for k, v in track_a.meta.items() if not isinstance(v, np.ndarray)})

        # -------------------------------------------------------------- S3 B
        self._begin("S3B", "Track B: phase-congruency structural matcher")
        track_b = self._run_track_b(b_src, b_ref, pre)
        self._end("S3B", "Track B: phase-congruency structural matcher",
                  n_matches=len(track_b), input=self.cfg.match.track_b_input,
                  **{k: v for k, v in track_b.meta.items() if not isinstance(v, np.ndarray)})

        # ---------------------------------------------------------- S3 gate
        self._begin("S3G", "Agreement gate")
        gate: GateResult = agreement_gate(
            track_a, track_b,
            tau_px=self.cfg.match.agreement_tau_px,
            sharpness_peak_ratio=self.cfg.match.sharpness_peak_ratio,
            sharpness_peak_min=self.cfg.match.sharpness_peak_min,
            src_repr=src_repr, ref_repr=ref_repr,
        )
        self._end("S3G", "Agreement gate", **gate.summary())

        # -------------------------------------------------------------- S4
        self._begin("S4", "Sub-pixel refinement and covariance")
        r_src, r_ref = inputs[self.cfg.refine.input]
        tiepoints, refine_report = refine_matches(
            gate.matches, r_src, r_ref,
            patch=self.cfg.refine.patch,
            upsample_factor=self.cfg.refine.upsample_factor,
            use_lsm=self.cfg.refine.lsm,
            lsm_max_iter=self.cfg.refine.lsm_max_iter,
            lsm_tol_px=self.cfg.refine.lsm_tol_px,
            max_sigma_px=self.cfg.refine.max_sigma_px,
            max_shift_px=self.cfg.refine.max_shift_px,
            covariance_method=self.cfg.refine.covariance,
            confidence=gate.confidence, origin=gate.origin,
        )
        self._end("S4", "Sub-pixel refinement and covariance",
                  input=self.cfg.refine.input, **refine_report.to_dict())

        # -------------------------------------------------------------- S5
        self._begin("S5", "Outlier rejection and model fit")
        emission_diff = abs(source.illum.emission_deg - reference.illum.emission_deg)
        global_fit = fit_global_auto(
            tiepoints,
            preference=self.cfg.model.global_model,
            emission_diff_deg=emission_diff,
            threshold_scale=self.cfg.model.adaptive_threshold_scale,
            floor_px=self.cfg.model.threshold_floor_px,
            ceiling_px=self.cfg.model.threshold_ceiling_px,
            homography_emission_deg=self.cfg.model.homography_emission_deg,
            similarity_min_inliers=self.cfg.model.similarity_min_inliers,
            confidence=self.cfg.model.ransac_confidence,
            max_iters=self.cfg.model.ransac_max_iters,
        ) if len(tiepoints) >= 4 else None

        local_model = jitter = None
        if global_fit is not None:
            mark_inliers(tiepoints, global_fit)
            inliers = [t for t in tiepoints if t.inlier]
            if len(inliers) >= 10:
                src_pts = np.array([[t.src_sample, t.src_line] for t in inliers])
                resid = np.array([[t.residual_x, t.residual_y] for t in inliers])
                local_model = fit_local(src_pts, -resid, self.cfg.model.local_model,
                                        self.cfg.model.tps_lambda_grid)
                if self.cfg.model.jitter_model:
                    jitter = fit_jitter(src_pts[:, 1], -resid, self.cfg.model.jitter_spline_knots)

        self._end("S5", "Outlier rejection and model fit",
                  **(global_fit.to_dict() if global_fit else {"fitted": False}),
                  local=local_model.to_dict() if local_model else None,
                  jitter=jitter.to_dict() if jitter else None,
                  emission_difference_deg=round(emission_diff, 3))

        # -------------------------------------------------------------- S6
        self._begin("S6", "Uniformity enforcement")
        lattice_size = (
            auto_lattice_size(self.cfg.uniform.target_points)
            if self.cfg.uniform.auto_lattice else tuple(self.cfg.uniform.lattice)
        )
        lattice = build_lattice(src_img.shape, lattice_size, pre.overlap)
        lattice.assign(tiepoints)

        uniformity_before = uniformity_report(tiepoints, lattice)
        kept = apply_quota(tiepoints, lattice, self.cfg.uniform.per_cell_quota, self.cfg.uniform.anms_radius_px)

        reseed_rep = None
        if self.cfg.uniform.reseed and global_fit is not None:
            kept, reseed_rep = reseed_empty_cells(
                kept, src_repr, ref_repr, lattice, global_fit.H,
                max_passes=self.cfg.uniform.reseed_passes,
                window_px=self.cfg.uniform.reseed_window_px,
                threshold_scale=self.cfg.uniform.reseed_threshold_scale,
            )
            if global_fit is not None:
                mark_inliers(kept, global_fit)

        lattice.assign(kept)
        uniformity_after = uniformity_report(kept, lattice)
        self._end("S6", "Uniformity enforcement",
                  before=uniformity_before, after=uniformity_after,
                  reseed=reseed_rep.to_dict() if reseed_rep else None)

        return {
            "tiepoints": kept,
            "all_tiepoints": tiepoints,
            "global_fit": global_fit,
            "local_model": local_model,
            "jitter": jitter,
            "gate": gate,
            "track_a": track_a,
            "track_b": track_b,
            "lattice": lattice,
            "uniformity": uniformity_after,
            "uniformity_before": uniformity_before,
            "render": render,
            "src_repr": src_repr,
            "ref_repr": ref_repr,
            "ref_for_matching": ref_for_matching,
            "refine": refine_report,
        }

    # ------------------------------------------------------------- the tracks

    def _run_track_a(self, src_repr: np.ndarray, ref_repr: np.ndarray, pre: PrealignResult) -> MatchSet:
        if not self.cfg.match.track_a:
            return MatchSet.empty("A", reason="track A disabled in configuration")

        matcher = build_matcher(
            self.cfg.match.deep_matcher, self.cfg.match.device,
            self.cfg.match.deep_weights_dir, self.cfg.match.deep_conf_threshold,
        )
        if matcher is None or not matcher.available():
            return MatchSet.empty("A", reason="no deep matcher available in this environment",
                                  status=track_a_status(self.cfg.match.device, self.cfg.match.deep_weights_dir))

        size = self.cfg.match.tile_size
        h, w = src_repr.shape[:2]
        if h <= size and w <= size:
            ms = matcher.match(src_repr, ref_repr)
            ms.meta.setdefault("tiles", 1)
            return ms

        parts: list[MatchSet] = []
        for src_tile, ref_tile, pair in iter_tiles(
            src_repr, ref_repr, size, self.cfg.match.tile_overlap, pre.residual_px
        ):
            ms = matcher.match(src_tile, ref_tile)
            if ms.is_empty:
                continue
            parts.append(ms.top(self.cfg.match.max_matches_per_tile)
                         .offset(pair.src_origin, pair.ref_origin))

        merged = MatchSet.concat(parts, "A")
        merged.meta["tiles"] = len(parts)
        merged.meta["matcher"] = matcher.name
        return merged.deduplicate(2.0)

    def _run_track_b(self, src_repr: np.ndarray, ref_repr: np.ndarray, pre: PrealignResult) -> MatchSet:
        if not self.cfg.match.track_b:
            return MatchSet.empty("B", reason="track B disabled in configuration")

        # The search window is derived from S1's residual estimate plus a safety factor,
        # never from a constant: that is what keeps a repeated crater out of contention.
        window = int(np.clip(pre.residual_px * 1.5 + 16, 24, 256))
        matcher = StructuralMatcher(
            lattice=tuple(self.cfg.match.lattice_detect),
            per_cell=self.cfg.match.detect_per_cell,
            patch=self.cfg.match.mim_patch,
            lowe_ratio=self.cfg.match.lowe_ratio,
            window=window,
            metric=self.cfg.match.template_metric,
            pc_k=self.cfg.illum.pc_k,
            pc_noise_adaptive=self.cfg.illum.pc_noise_adaptive,
        )
        return matcher.match(src_repr, ref_repr, mask=pre.overlap)

    # ----------------------------------------------------------------- metrics

    def _metrics(self, result, pre, source, reference, elapsed: float) -> dict[str, Any]:
        from setu.eval.metrics import ce90, match_density, runtime_per_megapixel

        tiepoints: list[TiePoint] = result["tiepoints"]
        inliers = [t for t in tiepoints if t.inlier]
        fit = result["global_fit"]
        gsd = pre.grid.gsd_m

        resid = np.array([t.residual_norm for t in inliers if np.isfinite(t.residual_norm)])
        rmse_px = float(np.sqrt(np.mean(resid**2))) if resid.size else float("nan")

        area_km2 = pre.overlap.sum() * (gsd**2) / 1e6

        sigmas = np.array([t.sigma_rms for t in inliers])
        sigmas = sigmas[np.isfinite(sigmas)]

        return {
            "n_tiepoints": len(tiepoints),
            "n_inliers": len(inliers),
            "inlier_ratio": fit.inlier_ratio if fit else 0.0,
            "putative_count": fit.n_input if fit else 0,
            "rmse_px": round(rmse_px, 4),
            "rmse_m": round(rmse_px * gsd, 4) if np.isfinite(rmse_px) else None,
            "note_on_rmse": (
                "This is the residual of the fitted global model on its own inliers, not "
                "an independent accuracy. Checkpoint and LOOCV RMSE are the accuracy figures."
            ),
            "ce90_px": round(float(np.percentile(resid, 90)), 4) if resid.size else None,
            "ce90_m": round(float(np.percentile(resid, 90)) * gsd, 4) if resid.size else None,
            "loocv_rmse_px": result["local_model"].loocv_rmse_px if result["local_model"] else None,
            "local_fit_rmse_px": result["local_model"].fit_rmse_px if result["local_model"] else None,
            "median_sigma_px": round(float(np.median(sigmas)), 4) if sigmas.size else None,
            "sigma_variance_factor": round(result["refine"].variance_factor, 4),
            "match_density_per_km2": round(match_density(len(inliers), area_km2), 3),
            "overlap_area_km2": round(float(area_km2), 4),
            "gsd_work_m": round(gsd, 4),
            "uniformity": result["uniformity"],
            "gate": result["gate"].summary(),
            "jitter_amplitude_px": result["jitter"].amplitude_px if result["jitter"] else None,
            "runtime_s": round(elapsed, 3),
            "runtime_s_per_megapixel": round(runtime_per_megapixel(elapsed, pre.src_ortho.shape), 4),
            "reillumination_applied": result["render"] is not None,
        }


def _render_at_native_resolution(
    dem: np.ndarray,
    dem_gsd_m: float | None,
    work_gsd_m: float,
    ref_img: np.ndarray,
    src_img: np.ndarray,
    illum,
    cfg,
    max_render_px: int = 2048,
) -> RenderResult:
    """Render at the terrain model's own resolution, then decimate to the working grid.

    The order matters and it is not obvious. Shading is a non-linear function of slope, so
    smoothing a DEM and rendering it does *not* give the same image as rendering it and
    then smoothing: sub-pixel roughness contributes to the mean radiance of a pixel, and a
    decimated DEM has thrown that roughness away. The rendered scene comes out too smooth,
    and at large decimation factors it stops resembling the real image - which is
    decimated *radiance*, not decimated terrain - closely enough to match against.

    Measured on the benchmark, this is the difference between a correlation of 0.70 and
    0.91 at an eightfold ratio, and between no registration and a working one.
    """
    import cv2

    d = np.asarray(dem, dtype=np.float32)
    native_gsd = float(dem_gsd_m or work_gsd_m)

    # Cap the render size: beyond a couple of thousand pixels the extra fidelity is below
    # what survives the decimation anyway, and the shadow sweep is the cost driver.
    if max(d.shape) > max_render_px:
        f = max(d.shape) / max_render_px
        d = cv2.resize(d, (int(d.shape[1] / f), int(d.shape[0] / f)), interpolation=cv2.INTER_AREA)
        native_gsd *= f

    # The reference at its own resolution, for the albedo estimate.
    ref_native = cv2.resize(ref_img, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_CUBIC)

    rendered = reilluminate_reference(
        d, native_gsd, illum,
        reference_image=ref_native,
        shadow_method=cfg.illum.shadow_method,
        psf_sigma_px=cfg.illum.psf_sigma_px,
        match_noise=cfg.illum.match_noise,
        source_image=src_img,
    )

    h, w = ref_img.shape[:2]
    if rendered.image.shape[:2] != (h, w):
        interp = cv2.INTER_AREA if rendered.image.shape[0] > h else cv2.INTER_CUBIC
        rendered.image = cv2.resize(rendered.image, (w, h), interpolation=interp)
        rendered.shadow = cv2.resize(
            rendered.shadow.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        rendered.meta["rendered_at_gsd_m"] = round(native_gsd, 4)
        rendered.meta["decimated_to_gsd_m"] = round(work_gsd_m, 4)
        rendered.meta["render_decimation"] = round(work_gsd_m / max(native_gsd, 1e-9), 3)
    return rendered


def _lift_to_product_frame(result: dict[str, Any], correction: np.ndarray) -> None:
    """Rescale the fitted transform and the tie points from the working grid to the product frame.

    Only the reference side moves: source coordinates are already in the source product's
    own pixels, because the source is never decimated below the working GSD - it is the
    coarser image by definition of how the working GSD is chosen.
    """
    fit = result.get("global_fit")
    if fit is not None:
        fit.H = correction @ fit.H
        fit.H = fit.H / fit.H[2, 2]
        fit.residuals = fit.residuals * float(correction[0, 0])
        fit.threshold_px = float(fit.threshold_px * correction[0, 0])

    scale = float(correction[0, 0])
    for t in result.get("tiepoints", []):
        t.ref_sample *= scale
        t.ref_line *= scale
        if np.isfinite(t.residual_x):
            t.residual_x *= scale
            t.residual_y *= scale


def _fit_dem_to(dem: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a DEM patch onto the working grid.

    The DEM handed to `Pipeline.run` must already cover the *same ground* as the
    reference raster; only its resolution may differ. Nothing here can detect a DEM of
    the wrong extent, and re-illuminating from the wrong piece of terrain produces a
    render that looks entirely plausible and matches nothing.
    """
    import cv2

    d = np.asarray(dem, dtype=np.float32)
    if d.shape[:2] == tuple(shape[:2]):
        return d
    return cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _pixel_to_map_correction(H_px: np.ndarray, grid) -> np.ndarray:
    """Lift a pixel-space transform into map space, for the S5 -> S1 feedback edge.

    The grid affine is north-up with a single scale, so the conjugation is exact rather
    than an approximation.
    """
    a, _, c, _, e, f = list(grid.transform)[:6]
    A = np.array([[a, 0.0, c], [0.0, e, f], [0.0, 0.0, 1.0]], dtype=np.float64)
    return A @ np.asarray(H_px, dtype=np.float64) @ np.linalg.inv(A)
