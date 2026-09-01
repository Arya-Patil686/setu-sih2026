"""Section 7.1 - every metric the evaluation protocol requires.

Two rules govern this module. Errors are reported in the units the problem statement
names - source pixels *and* metres, always both. And a fitted model's own residual is
never presented as accuracy: `rmse_vs_truth` and `rmse_fit` are separate functions with
separate names, because conflating them is the most common way a registration result is
overstated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats
from shapely.geometry import MultiPoint, Polygon

# --------------------------------------------------------------- basic error

def residuals(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-point radial error between predicted and true positions."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
    t = np.asarray(truth, dtype=np.float64).reshape(-1, 2)
    return np.hypot(*(p - t).T)


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    """sqrt( (1/N) * sum ||p_i - t_i||^2 )."""
    r = residuals(pred, truth)
    r = r[np.isfinite(r)]
    return float(np.sqrt(np.mean(r**2))) if r.size else float("nan")


def rmse_vs_truth(H_est: np.ndarray, H_true: np.ndarray, shape: tuple[int, int], n: int = 24) -> float:
    """True geometric RMSE, in source pixels.

    Both transforms are applied to the same dense grid of source pixels and the
    disagreement is measured directly. Nothing here depends on which points the
    estimator happened to find, so a method cannot improve this number by discarding
    the correspondences it found hardest.
    """
    from setu.bench.generate import apply_h

    if H_est is None or not np.all(np.isfinite(H_est)):
        return float("inf")
    h, w = shape[:2]
    xs = np.linspace(0.05 * w, 0.95 * w, n)
    ys = np.linspace(0.05 * h, 0.95 * h, n)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    return rmse(apply_h(H_est, pts), apply_h(H_true, pts))


def rmse_fit(src: np.ndarray, ref: np.ndarray, H: np.ndarray) -> float:
    """Residual of the fitted model on its own tie points.

    Reported for completeness and clearly labelled. It is not accuracy: a flexible local
    model can drive this to nearly zero while being worthless out of sample, which is
    why `loocv_rmse` sits beside it in every table.
    """
    from setu.bench.generate import apply_h

    return rmse(apply_h(H, src), ref)


def ce90(pred: np.ndarray, truth: np.ndarray, percentile: float = 90.0) -> float:
    """CE90 - the 90th percentile of radial error, the geodetic accuracy statement."""
    r = residuals(pred, truth)
    r = r[np.isfinite(r)]
    return float(np.percentile(r, percentile)) if r.size else float("nan")


def px_to_m(value_px: float, gsd_m: float) -> float:
    """Errors are quoted in both units throughout; this is the only conversion used."""
    return float(value_px * gsd_m)


def success_rate(rmse_values: Sequence[float], thresholds: Sequence[float]) -> dict[str, float]:
    """SR@t - the fraction of pairs registered to within t pixels."""
    arr = np.asarray([v for v in rmse_values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {f"SR@{t:g}px": float("nan") for t in thresholds}
    return {f"SR@{t:g}px": float((arr <= t).mean()) for t in thresholds}


def inlier_stats(n_inliers: int, n_putative: int) -> dict[str, float]:
    return {
        "inlier_count": int(n_inliers),
        "putative_count": int(n_putative),
        "inlier_ratio": float(n_inliers / n_putative) if n_putative else 0.0,
    }


def match_density(n_inliers: int, area_km2: float) -> float:
    """Inliers per square kilometre - comparable across scales, unlike a raw count."""
    return float(n_inliers / area_km2) if area_km2 > 0 else float("nan")


# ------------------------------------------------------------- S6 uniformity

def coverage_ratio(points: np.ndarray, shape: tuple[int, int], lattice: tuple[int, int]) -> float:
    """Fraction of lattice cells containing at least one point. Target >= 0.90."""
    occ = cell_occupancy(points, shape, lattice)
    return float((occ > 0).mean()) if occ.size else 0.0


def cell_occupancy(points: np.ndarray, shape: tuple[int, int], lattice: tuple[int, int]) -> np.ndarray:
    """Point count per lattice cell, flattened."""
    m, n = lattice
    h, w = shape[:2]
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if p.size == 0:
        return np.zeros(m * n, dtype=int)
    col = np.clip((p[:, 0] / max(w, 1e-9) * n).astype(int), 0, n - 1)
    row = np.clip((p[:, 1] / max(h, 1e-9) * m).astype(int), 0, m - 1)
    return np.bincount(row * n + col, minlength=m * n)


def occupancy_chi_square(points: np.ndarray, shape: tuple[int, int], lattice: tuple[int, int]) -> dict[str, float]:
    """chi^2 = sum over cells of (o_c - e)^2 / e, against the uniform null.

    A high p-value means the occupancy is statistically indistinguishable from uniform,
    which is what the problem statement's uniform-distribution requirement asks for.
    Reporting the p-value rather than the raw chi-square is what makes the number
    comparable between runs with different point counts.
    """
    occ = cell_occupancy(points, shape, lattice)
    n_cells = occ.size
    total = int(occ.sum())
    if total == 0 or n_cells < 2:
        return {"chi2": float("nan"), "chi2_p": float("nan"), "dof": n_cells - 1}
    expected = total / n_cells
    chi2 = float(((occ - expected) ** 2 / expected).sum())
    dof = n_cells - 1
    return {"chi2": chi2, "chi2_p": float(stats.chi2.sf(chi2, dof)), "dof": dof}


def clark_evans(points: np.ndarray, area: float) -> dict[str, float]:
    """Clark-Evans nearest-neighbour index R, per the Appendix.

    R = 1 for a random (Poisson) process, R > 1 for a dispersed pattern, R < 1 for a
    clustered one. The target band is 1.0 to 1.4: SETU wants points spread more evenly
    than chance, but forcing R much above 1.4 would mean placing points on a rigid grid
    regardless of whether there is any matchable texture there.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(p)
    if n < 3 or area <= 0:
        return {"clark_evans_R": float("nan"), "clark_evans_z": float("nan"), "n": n}

    from scipy.spatial import cKDTree

    tree = cKDTree(p)
    d, _ = tree.query(p, k=2)
    r_obs = float(d[:, 1].mean())
    r_exp = 0.5 * np.sqrt(area / n)
    R = r_obs / r_exp if r_exp > 0 else float("nan")
    sigma_r = 0.26136 / np.sqrt(n * n / area)
    z = (r_obs - r_exp) / sigma_r if sigma_r > 0 else float("nan")
    return {"clark_evans_R": float(R), "clark_evans_z": float(z), "r_obs_px": r_obs, "r_exp_px": float(r_exp), "n": n}


def uniformity(
    points: np.ndarray,
    shape: tuple[int, int],
    lattice: tuple[int, int] = (8, 8),
    overlap_area_px: float | None = None,
) -> dict[str, float]:
    """The three uniformity statistics of S6, together."""
    h, w = shape[:2]
    area = overlap_area_px if overlap_area_px is not None else float(h * w)
    out: dict[str, float] = {
        "coverage_ratio": coverage_ratio(points, shape, lattice),
        "lattice": list(lattice),
        "n_points": int(np.asarray(points).reshape(-1, 2).shape[0]),
    }
    out.update(occupancy_chi_square(points, shape, lattice))
    out.update(clark_evans(points, area))
    return out


def overlap_polygon_area(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """True overlap area in pixels - the intersection of validity masks, not bounding boxes."""
    return float((np.asarray(mask_a, bool) & np.asarray(mask_b, bool)).sum())


# --------------------------------------------------------- N4 sigma calibration

def sigma_calibration(sigmas: Sequence[float], residual_norms: Sequence[float], n_bins: int = 8) -> dict[str, Any]:
    """Predicted sigma against realised residual - the N4 calibration curve.

    A well-calibrated uncertainty puts this curve on y = x. Points are binned by
    predicted sigma and the realised RMS residual is measured per bin, so the curve
    shows *where* the prediction breaks down rather than collapsing it to one number.
    """
    s = np.asarray(sigmas, dtype=np.float64)
    r = np.asarray(residual_norms, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(r) & (s > 0)
    s, r = s[ok], r[ok]
    if s.size < 8:
        return {"calibration_error": float("nan"), "n": int(s.size), "curve": []}

    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    curve = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (s >= lo) & (s <= hi)
        if sel.sum() < 3:
            continue
        curve.append({
            "sigma_pred": float(s[sel].mean()),
            "residual_rms": float(np.sqrt(np.mean(r[sel] ** 2))),
            "n": int(sel.sum()),
        })
    err = float(np.mean([abs(c["sigma_pred"] - c["residual_rms"]) for c in curve])) if curve else float("nan")
    return {"calibration_error": err, "n": int(s.size), "curve": curve}


# ------------------------------------------------------- validation surrogates

def loocv_rmse(src: np.ndarray, ref: np.ndarray, fit_fn: Callable[..., Any], max_points: int = 300) -> float:
    """Leave-one-out cross-validated RMSE over the automatic tie points.

    This is the guard against an over-flexible local model. A thin-plate spline with too
    little regularisation reports a fit RMSE near zero and is worthless; that same spline
    scores badly here, which is the entire point of computing it.
    """
    from setu.bench.generate import apply_h

    s = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    r = np.asarray(ref, dtype=np.float64).reshape(-1, 2)
    n = len(s)
    if n < 8:
        return float("nan")

    idx = np.arange(n)
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)

    errs = []
    for i in idx:
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        try:
            model = fit_fn(s[keep], r[keep])
        except Exception:
            continue
        pred = model(s[i:i + 1]) if callable(model) else apply_h(model, s[i:i + 1])
        errs.append(float(np.hypot(*(pred.reshape(2) - r[i]))))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def cross_consistency(H_ab: np.ndarray, H_ba: np.ndarray) -> float:
    """|| H_AB . H_BA - I ||_F, normalised.

    Registering A to B and B to A independently should compose to the identity. Any
    departure is systematic bias, and it is detectable with no external truth at all,
    which makes it the one honest check available on a real cross-sensor pair.
    """
    if H_ab is None or H_ba is None:
        return float("nan")
    try:
        comp = np.asarray(H_ab, float) @ np.asarray(H_ba, float)
        comp = comp / comp[2, 2]
    except Exception:
        return float("nan")
    return float(np.linalg.norm(comp - np.eye(3), ord="fro"))


# ----------------------------------------------------------------- bootstrap

@dataclass
class Interval:
    """A statistic with its bootstrap confidence interval."""

    value: float
    lo: float
    hi: float
    n: int

    def to_dict(self) -> dict[str, float]:
        return {"value": self.value, "ci_lo": self.lo, "ci_hi": self.hi, "n": self.n}

    def __repr__(self) -> str:
        return f"{self.value:.4g} [{self.lo:.4g}, {self.hi:.4g}] (n={self.n})"


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 26166,
) -> Interval:
    """Resample pairs with replacement and report the percentile interval.

    Applied to every headline number. A single figure with no interval invites the
    question "on how many images?", and on a benchmark this small the honest answer is
    usually that the interval is wide.
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    if arr.size == 1:
        v = float(statistic(arr))
        return Interval(v, v, v, 1)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    stats_boot = np.array([statistic(arr[i]) for i in idx])
    return Interval(
        value=float(statistic(arr)),
        lo=float(np.percentile(stats_boot, 100 * alpha / 2)),
        hi=float(np.percentile(stats_boot, 100 * (1 - alpha / 2))),
        n=int(arr.size),
    )


def summarise(values_by_metric: dict[str, Sequence[float]], n_boot: int = 1000, alpha: float = 0.05) -> dict[str, Any]:
    """Bootstrap every metric in a dictionary of per-pair values."""
    return {k: bootstrap_ci(v, n_boot=n_boot, alpha=alpha).to_dict() for k, v in values_by_metric.items()}


def runtime_per_megapixel(seconds: float, shape: tuple[int, int]) -> float:
    mp = (shape[0] * shape[1]) / 1e6
    return float(seconds / mp) if mp > 0 else float("nan")


def points_to_polygon(points: np.ndarray) -> Polygon:
    """Convex hull of a point set, used for area when no mask is available."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return MultiPoint([tuple(x) for x in p]).convex_hull if len(p) >= 3 else Polygon()
