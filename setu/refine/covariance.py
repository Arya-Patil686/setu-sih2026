"""S4 - per-correspondence uncertainty. Novelty N4.

Every tie point carries a 2 x 2 covariance. Those covariances weight the robust
geometric fit, set the outlier threshold adaptively, and are written per point into the
output file. That is what turns a match list into a product an agency can ingest, and it
is the claim validated by the calibration curve: predicted sigma against realised
residual should lie on y = x.

Getting there needs one correction that the textbook formula omits. The a-posteriori
variance sigma0^2 = r'r / (N - u) assumes independent observations. Image residuals are
not independent: bicubic resampling and the sensor point-spread function correlate
neighbouring pixels over several pixels, so the true number of independent samples in a
64 x 64 patch is a small fraction of 4096. Left uncorrected, the reported sigma comes out
roughly an order of magnitude too small, and every downstream threshold derived from it
is then wrong in the same direction.
"""

from __future__ import annotations

import numpy as np


def effective_sample_size(residual: np.ndarray) -> float:
    """Independent-sample count of a 2-D residual field, from its lag-1 autocorrelation.

    Treating each axis as a first-order autoregressive process gives the standard
    variance-inflation factor (1 - rho) / (1 + rho) per axis. It is a coarse model of a
    complicated correlation structure, but it is unbiased in the direction that matters
    and it needs no tuning constants.
    """
    r = np.asarray(residual, dtype=np.float64)
    if r.ndim == 1 or min(r.shape) < 4:
        return float(r.size)
    r = r - r.mean()
    var = float((r * r).mean())
    if var < 1e-20:
        return float(r.size)

    rho_y = float((r[:-1, :] * r[1:, :]).mean() / var)
    rho_x = float((r[:, :-1] * r[:, 1:]).mean() / var)
    rho_y = float(np.clip(rho_y, 0.0, 0.99))
    rho_x = float(np.clip(rho_x, 0.0, 0.99))

    factor = ((1 - rho_y) / (1 + rho_y)) * ((1 - rho_x) / (1 + rho_x))
    return float(max(r.size * factor, 16.0))


def covariance_from_lsm(
    normal_matrix: np.ndarray,
    residual: np.ndarray,
    n_params: int = 8,
    translation_idx: tuple[int, int] = (2, 5),
) -> np.ndarray:
    """Sigma = sigma0^2 * N^-1, restricted to the two translation parameters.

    `residual` is the full residual image from the final LSM iteration, used both for
    the sum of squares and for the effective sample size.
    """
    r = np.asarray(residual, dtype=np.float64)
    n_eff = effective_sample_size(r)
    dof = max(n_eff - n_params, 1.0)
    # The sum of squares is scaled to the effective sample count so that sigma0^2 remains
    # a per-independent-observation variance.
    sigma0_sq = float((r.ravel() @ r.ravel()) / r.size * n_eff / dof)

    try:
        cov_full = sigma0_sq * np.linalg.inv(np.asarray(normal_matrix, dtype=np.float64))
    except np.linalg.LinAlgError:
        return np.full((2, 2), np.nan)

    idx = np.asarray(translation_idx)
    cov = cov_full[np.ix_(idx, idx)]
    return cov if np.isfinite(cov).all() else np.full((2, 2), np.nan)


def covariance_from_paraboloid(surface: np.ndarray, peak: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Forstner covariance from the curvature of the correlation surface.

    Fits c(x, y) = c0 + b'x + 0.5 x'Hx on the 3 x 3 neighbourhood of the peak. The
    sub-pixel offset is delta = -H^-1 b and the covariance is Sigma = sigma_n^2 * H^-1,
    with sigma_n^2 taken from the residual of the paraboloid fit.

    Cheaper than LSM and available for points that only ever went through phase
    correlation, which is why both routes exist.
    """
    s = np.asarray(surface, dtype=np.float64)
    r, c = peak
    if r < 1 or c < 1 or r >= s.shape[0] - 1 or c >= s.shape[1] - 1:
        return np.zeros(2), np.full((2, 2), np.nan)

    patch = s[r - 1:r + 2, c - 1:c + 2]
    yy, xx = np.mgrid[-1:2, -1:2]
    # Design matrix for the full quadratic: [1, x, y, x^2/2, y^2/2, xy].
    A = np.column_stack([
        np.ones(9), xx.ravel(), yy.ravel(),
        0.5 * xx.ravel() ** 2, 0.5 * yy.ravel() ** 2, (xx * yy).ravel(),
    ])
    try:
        coef, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(2), np.full((2, 2), np.nan)

    b = np.array([coef[1], coef[2]])
    H = np.array([[coef[3], coef[5]], [coef[5], coef[4]]])

    # A maximum needs a negative-definite Hessian; anything else is a saddle or a ridge,
    # which means the peak is not localised and no covariance should be reported.
    eigs = np.linalg.eigvalsh(H)
    if np.any(eigs >= -1e-12):
        return np.zeros(2), np.full((2, 2), np.nan)

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.zeros(2), np.full((2, 2), np.nan)

    delta = -H_inv @ b
    resid = patch.ravel() - A @ coef
    sigma_n_sq = float(resid @ resid / max(9 - 6, 1))
    cov = sigma_n_sq * (-H_inv)          # -H^-1 is positive definite at a maximum
    return np.clip(delta, -1.0, 1.0), cov


def covariance_to_fields(cov: np.ndarray) -> tuple[float, float, float]:
    """(sigma_x, sigma_y, sigma_xy) as written to the tie-point file."""
    if cov is None or not np.isfinite(cov).all():
        return float("nan"), float("nan"), 0.0
    vx, vy = float(cov[0, 0]), float(cov[1, 1])
    if vx < 0 or vy < 0:
        return float("nan"), float("nan"), 0.0
    return float(np.sqrt(vx)), float(np.sqrt(vy)), float(cov[0, 1])


def error_ellipse(cov: np.ndarray, n_sigma: float = 1.0) -> dict[str, float]:
    """Semi-axes and orientation of the error ellipse, for plotting."""
    if cov is None or not np.isfinite(cov).all():
        return {"a": float("nan"), "b": float("nan"), "theta_deg": float("nan")}
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    vals = np.clip(vals, 0.0, None)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return {
        "a": float(n_sigma * np.sqrt(vals[0])),
        "b": float(n_sigma * np.sqrt(vals[1])),
        "theta_deg": float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))),
    }


def variance_component_scale(
    sigmas: np.ndarray,
    residual_norms: np.ndarray,
    trim: float = 0.2,
    lo: float = 0.2,
    hi: float = 50.0,
) -> float:
    """A-posteriori variance factor, estimated from the run's own residuals.

    A covariance built from image noise alone is systematically optimistic, and by a
    fairly stable factor. It accounts for photon and read noise but not for the errors
    that actually dominate a sub-pixel match: resampling error, the linearisation of the
    least-squares model, a DEM that is coarser than the imagery, and the small
    unmodelled geometric distortion the global transform cannot absorb.

    Rather than hard-coding an inflation constant, the factor is estimated the way a
    geodetic network estimates its variance of unit weight - from the agreement between
    predicted and realised errors on this run's own points. It is written into
    `metrics.json` as `sigma_variance_factor`, so a reader can see exactly how much the
    raw covariances were inflated and judge the result accordingly.

    A trimmed ratio is used because outliers would otherwise inflate the factor for
    every point in the run.
    """
    s = np.asarray(sigmas, dtype=np.float64)
    r = np.asarray(residual_norms, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(r) & (s > 1e-9)
    s, r = s[ok], r[ok]
    if s.size < 12:
        return 1.0

    ratio = r / s
    k = int(trim * ratio.size)
    trimmed = np.sort(ratio)[k:ratio.size - k] if ratio.size - 2 * k >= 8 else ratio
    # The RMS of the ratio, not its mean: the factor multiplies a standard deviation, so
    # it has to be consistent in the variance domain.
    scale = float(np.sqrt(np.mean(trimmed**2)))
    return float(np.clip(scale, lo, hi))


def apply_variance_scale(cov: np.ndarray, scale: float) -> np.ndarray:
    """Inflate a covariance by a variance factor (which scales standard deviations)."""
    if cov is None or not np.isfinite(cov).all():
        return cov
    return np.asarray(cov, dtype=np.float64) * (scale**2)
