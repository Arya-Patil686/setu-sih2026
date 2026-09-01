"""Section 7 - the evaluation harness.

Runs every method over a benchmark set under identical conditions and measures each one
against exact ground truth. Two rules make the comparison fair, and both are enforced
here rather than left to the caller:

  * Every method - baseline, ablation and the full system - goes through the same robust
    fit and the same metric code. Only the correspondence step differs.
  * Accuracy is always measured against `H_true`, never against the model that was just
    fitted to the data. Both numbers are reported and they are given different names.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from setu.bench.generate import BenchPair, apply_h
from setu.config import SetuConfig
from setu.eval.baselines import ABLATIONS, ABLATION_LABELS, build_baseline
from setu.eval.metrics import (
    bootstrap_ci,
    ce90,
    clark_evans,
    coverage_ratio,
    match_density,
    occupancy_chi_square,
    residuals,
    rmse,
    rmse_vs_truth,
    success_rate,
)
from setu.match.base import MatchSet
from setu.model.robust import GlobalFit, fit_global
from setu.types import TiePoint


@dataclass
class PairResult:
    """One method on one pair."""

    method: str
    pair_id: str
    ok: bool
    n_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    #: True geometric error of the estimated transform, from exact truth.
    rmse_true_px: float = float("nan")
    #: Per-correspondence error against truth. This is the honest per-tie-point accuracy;
    #: `rmse_true_px` is the model's accuracy and is smaller by roughly sqrt(N).
    rmse_points_px: float = float("nan")
    #: Per-point error over the *inliers* only. This is the like-for-like comparison:
    #: SETU emits an already-gated set, whereas a raw matcher's product would be
    #: RANSAC-filtered before anyone used it, so comparing SETU's gated points against
    #: another method's ungated ones measures the gate twice.
    rmse_inliers_px: float = float("nan")
    median_inlier_err_px: float = float("nan")
    #: Coverage and dispersion after subsampling every method to the same point count.
    #: Section N3 asks for exactly this: uniformity "at matched inlier counts".
    coverage_matched: float = float("nan")
    clark_evans_matched: float = float("nan")
    median_point_err_px: float = float("nan")
    precision_3px: float = float("nan")
    ce90_px: float = float("nan")
    rmse_fit_px: float = float("nan")
    coverage: float = float("nan")
    clark_evans_R: float = float("nan")
    chi2_p: float = float("nan")
    match_density_km2: float = float("nan")
    median_sigma_px: float = float("nan")
    seconds: float = float("nan")
    d_sun_elev: float = float("nan")
    d_sun_az: float = float("nan")
    scale_ratio: float = float("nan")
    gsd_src_m: float = float("nan")
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


def evaluate_matches(
    matches: MatchSet,
    pair: BenchPair,
    lattice: tuple[int, int] = (8, 8),
    tol_px: float = 3.0,
    fit_threshold_px: float = 3.0,
    sigmas: np.ndarray | None = None,
    matched_count: int = 150,
) -> dict[str, Any]:
    """Score one method's correspondences against a pair's exact ground truth."""
    shape = pair.source.shape[:2]
    out: dict[str, Any] = {"n_matches": len(matches)}
    if matches.is_empty:
        return out

    err = residuals(matches.kpts_ref, apply_h(pair.H_true, matches.kpts_src))
    out["rmse_points_px"] = float(np.sqrt(np.mean(err**2)))
    out["median_point_err_px"] = float(np.median(err))
    out["precision_3px"] = float((err <= tol_px).mean())
    out["ce90_px"] = float(np.percentile(err, 90))

    tiepoints = [
        TiePoint(
            tid=i,
            src_sample=float(matches.kpts_src[i, 0]), src_line=float(matches.kpts_src[i, 1]),
            ref_sample=float(matches.kpts_ref[i, 0]), ref_line=float(matches.kpts_ref[i, 1]),
            conf=float(matches.conf[i]),
            sigma_x=float(sigmas[i]) if sigmas is not None and i < len(sigmas) else np.nan,
            sigma_y=float(sigmas[i]) if sigmas is not None and i < len(sigmas) else np.nan,
        )
        for i in range(len(matches))
    ]

    fit: GlobalFit | None = fit_global(tiepoints, "affine", fit_threshold_px) if len(tiepoints) >= 4 else None
    if fit is None:
        out["n_inliers"] = 0
        out["inlier_ratio"] = 0.0
        return out

    inlier_pts = matches.kpts_src[fit.inliers]
    out["n_inliers"] = fit.n_inliers
    out["inlier_ratio"] = fit.inlier_ratio
    out["rmse_fit_px"] = float(np.sqrt(np.mean(fit.residuals[fit.inliers] ** 2))) if fit.n_inliers else float("nan")
    out["rmse_true_px"] = rmse_vs_truth(fit.H, pair.H_true, shape)
    out["coverage"] = coverage_ratio(inlier_pts, shape, lattice)
    out["clark_evans_R"] = clark_evans(inlier_pts, float(shape[0] * shape[1]))["clark_evans_R"]
    out["chi2_p"] = occupancy_chi_square(inlier_pts, shape, lattice)["chi2_p"]

    area_km2 = (shape[0] * pair.gsd_src_m) * (shape[1] * pair.gsd_src_m) / 1e6
    out["match_density_km2"] = match_density(fit.n_inliers, area_km2)

    inlier_err = err[fit.inliers]
    if inlier_err.size:
        out["rmse_inliers_px"] = float(np.sqrt(np.mean(inlier_err**2)))
        out["median_inlier_err_px"] = float(np.median(inlier_err))

    # Uniformity at a matched point count. A semi-dense matcher returning 3000 points
    # will cover every cell of an 8x8 lattice whatever its spatial behaviour, so
    # comparing raw coverage rewards density rather than distribution. Subsampling every
    # method to the same count is what makes the comparison about spread.
    if fit.n_inliers >= matched_count:
        idx = np.random.default_rng(26166).choice(fit.n_inliers, matched_count, replace=False)
        sub = inlier_pts[idx]
        out["coverage_matched"] = coverage_ratio(sub, shape, lattice)
        out["clark_evans_matched"] = clark_evans(sub, float(shape[0] * shape[1]))["clark_evans_R"]
    else:
        out["coverage_matched"] = out["coverage"]
        out["clark_evans_matched"] = out["clark_evans_R"]

    out["H_est"] = fit.H
    return out


def run_baseline_on_pair(method: str, pair: BenchPair, matcher_cache: dict, **kw: Any) -> PairResult:
    """Run one non-SETU baseline on one pair."""
    res = PairResult(method=method, pair_id=pair.pair_id, ok=False,
                     d_sun_elev=pair.d_sun_elev, d_sun_az=pair.d_sun_az,
                     scale_ratio=pair.scale_ratio, gsd_src_m=pair.gsd_src_m)
    try:
        if method not in matcher_cache:
            matcher_cache[method] = build_baseline(method)
        matcher = matcher_cache[method]
        if matcher is None or not matcher.available():
            res.error = f"{method} is not available in this environment"
            return res

        t0 = time.perf_counter()
        matches = matcher.match(pair.source, pair.reference)
        res.seconds = time.perf_counter() - t0

        scored = evaluate_matches(matches, pair, **kw)
        _apply(res, scored)
        res.ok = True
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        res.extra["traceback"] = traceback.format_exc(limit=3)
    return res


def run_setu_on_pair(
    variant: str,
    pair: BenchPair,
    base_config: SetuConfig,
    use_dem: bool = True,
    **kw: Any,
) -> PairResult:
    """Run SETU, or one of its ablations, on one pair."""
    from setu.pipeline import Pipeline

    res = PairResult(method=variant, pair_id=pair.pair_id, ok=False,
                     d_sun_elev=pair.d_sun_elev, d_sun_az=pair.d_sun_az,
                     scale_ratio=pair.scale_ratio, gsd_src_m=pair.gsd_src_m)
    try:
        cfg = _with_overrides(base_config, ABLATIONS.get(variant, {}))
        src, ref = pair.to_products()

        t0 = time.perf_counter()
        run = Pipeline(cfg).run(
            src, ref,
            dem=pair.dem_ref if use_dem else None,
            dem_gsd_m=pair.gsd_ref_m if use_dem else None,
            run_id=f"{variant}__{pair.pair_id}", synthetic=True,
        )
        res.seconds = time.perf_counter() - t0

        inliers = [t for t in run.tiepoints if t.inlier]
        if not inliers:
            res.error = "no inlier tie points survived the pipeline"
            return res

        src_pts = np.array([[t.src_sample, t.src_line] for t in inliers])
        ref_pts = np.array([[t.ref_sample, t.ref_line] for t in inliers])
        sig = np.array([t.sigma_rms for t in inliers])

        ms = MatchSet(src_pts, ref_pts, np.array([t.conf for t in inliers]))
        scored = evaluate_matches(ms, pair, sigmas=sig, **kw)
        _apply(res, scored)

        gm = run.transform.get("global")
        if gm:
            res.rmse_true_px = rmse_vs_truth(np.array(gm["matrix"]), pair.H_true, pair.source.shape)
        res.median_sigma_px = float(np.nanmedian(sig)) if np.isfinite(sig).any() else float("nan")
        res.extra["uniformity"] = run.metrics.get("uniformity", {})
        res.extra["gate"] = run.metrics.get("gate", {})
        res.extra["sigma_variance_factor"] = run.metrics.get("sigma_variance_factor")
        res.extra["reillumination_applied"] = run.metrics.get("reillumination_applied")
        res.ok = True
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        res.extra["traceback"] = traceback.format_exc(limit=3)
    return res


def _apply(res: PairResult, scored: dict[str, Any]) -> None:
    for key, value in scored.items():
        if key == "H_est":
            continue
        if hasattr(res, key):
            setattr(res, key, value)
        else:
            res.extra[key] = value


def _with_overrides(config: SetuConfig, overrides: dict[str, Any]) -> SetuConfig:
    """Apply dotted-path overrides to a copy of the configuration."""
    cfg = config.model_copy(deep=True)
    for path, value in overrides.items():
        target = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return cfg


def run_suite(
    pairs: Sequence[BenchPair],
    methods: Iterable[str],
    base_config: SetuConfig,
    progress: Callable[[str, str, int, int], None] | None = None,
    lattice: tuple[int, int] = (8, 8),
) -> list[PairResult]:
    """Run every method over every pair. Returns one row per (method, pair)."""
    methods = list(methods)
    results: list[PairResult] = []
    cache: dict[str, Any] = {}
    total = len(methods) * len(pairs)
    done = 0

    for method in methods:
        for pair in pairs:
            if method in ABLATIONS:
                r = run_setu_on_pair(method, pair, base_config,
                                     use_dem=ABLATIONS[method].get("illum.reilluminate", True),
                                     lattice=lattice)
            else:
                r = run_baseline_on_pair(method, pair, cache, lattice=lattice)
            results.append(r)
            done += 1
            if progress:
                progress(method, pair.pair_id, done, total)
    return results


def aggregate(
    results: Sequence[PairResult],
    n_boot: int = 1000,
    alpha: float = 0.05,
    thresholds: Sequence[float] = (1.0, 3.0, 5.0, 10.0),
) -> dict[str, dict[str, Any]]:
    """Aggregate per-pair rows into one summary per method, with bootstrap intervals.

    A single number with no interval invites the question "on how many images?", and on a
    benchmark this size the honest answer is usually that the interval is wide.
    """
    by_method: dict[str, list[PairResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    summary: dict[str, dict[str, Any]] = {}
    for method, rows in by_method.items():
        ok = [r for r in rows if r.ok]
        n_failed = len(rows) - len(ok)

        def col(attr: str) -> list[float]:
            return [getattr(r, attr) for r in ok if np.isfinite(getattr(r, attr))]

        entry: dict[str, Any] = {
            "n_pairs": len(rows),
            "n_ok": len(ok),
            "n_failed": n_failed,
            "label": ABLATION_LABELS.get(method, method),
        }
        for attr in ("rmse_true_px", "rmse_points_px", "rmse_inliers_px", "median_inlier_err_px",
                     "median_point_err_px", "precision_3px",
                     "inlier_ratio", "n_inliers", "n_matches", "coverage", "clark_evans_R",
                     "coverage_matched", "clark_evans_matched",
                     "ce90_px", "rmse_fit_px", "match_density_km2", "median_sigma_px", "seconds"):
            entry[attr] = bootstrap_ci(col(attr), n_boot=n_boot, alpha=alpha).to_dict()

        entry["success_rate"] = success_rate(col("rmse_true_px"), thresholds)
        entry["errors"] = sorted({r.error for r in rows if r.error})[:3]
        summary[method] = entry
    return summary
