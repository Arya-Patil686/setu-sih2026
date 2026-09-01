"""S4 step 2 - least-squares matching (Forstner).

Iteratively solves for the six affine and two radiometric parameters of the source patch
that minimise the sum of squared differences against the reference patch. It typically
converges in three to six iterations and gives both a sub-pixel shift and, more
importantly, the normal-equation matrix from which the per-point covariance of N4 comes.

Two radiometric parameters are not optional here. Even after re-illumination the two
images differ in gain and offset - different detectors, different exposure, a residual
albedo difference - and a purely geometric LSM would absorb that brightness difference
into a spurious shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class LSMResult:
    """Outcome of one least-squares match."""

    dx: float
    dy: float
    affine: np.ndarray          # 2 x 3, source-to-reference within the patch
    normal_matrix: np.ndarray   # 8 x 8 normal equations
    sigma0_sq: float            # a-posteriori variance of unit weight
    n_iter: int
    converged: bool
    rms_residual: float

    def translation_covariance(self) -> np.ndarray:
        """Sigma = sigma0^2 * N^-1, restricted to the two translation parameters.

        The full inverse is taken before restricting rather than inverting a 2x2 block,
        because the translation parameters are correlated with the affine and radiometric
        ones and ignoring that correlation understates the uncertainty.
        """
        try:
            cov = self.sigma0_sq * np.linalg.inv(self.normal_matrix)
        except np.linalg.LinAlgError:
            return np.full((2, 2), np.nan)
        idx = np.array([2, 5])            # translation terms of the affine parameter vector
        return cov[np.ix_(idx, idx)]


def _sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bicubic resampling at arbitrary coordinates."""
    return map_coordinates(image, [ys, xs], order=3, mode="reflect").astype(np.float64)


def least_squares_match(
    src_patch: np.ndarray,
    ref_patch: np.ndarray,
    max_iter: int = 12,
    tol_px: float = 1e-3,
    damping: float = 1e-6,
) -> LSMResult | None:
    """Solve for the affine and radiometric parameters aligning src to ref.

    The parameter vector is [a11, a12, tx, a21, a22, ty, gain, offset]. Design-matrix
    rows come from the chain rule on the reference gradient, which is the standard
    Forstner formulation.
    """
    ref = np.asarray(ref_patch, dtype=np.float64)
    src = np.asarray(src_patch, dtype=np.float64)
    if ref.shape != src.shape or min(ref.shape) < 9:
        return None
    if ref.std() < 1e-8 or src.std() < 1e-8:
        return None

    h, w = ref.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xc, yc = xx - cx, yy - cy

    # Start from the identity geometry and the gain/offset that match the two patches'
    # first two moments, which removes most of the radiometric difference immediately.
    gain = float(ref.std() / max(src.std(), 1e-8))
    offset = float(ref.mean() - gain * src.mean())
    p = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, gain, offset], dtype=np.float64)

    gy, gx = np.gradient(ref)
    gx_f, gy_f = gx.ravel(), gy.ravel()
    xc_f, yc_f = xc.ravel(), yc.ravel()

    N = np.zeros((8, 8))
    sigma0_sq = np.nan
    converged = False
    it = 0
    rms = np.nan

    for it in range(1, max_iter + 1):
        sx = p[0] * xc + p[1] * yc + p[2] + cx
        sy = p[3] * xc + p[4] * yc + p[5] + cy
        if not np.isfinite(sx).all() or not np.isfinite(sy).all():
            return None
        warped = _sample(src, sx.ravel(), sy.ravel()).reshape(h, w)
        model = p[6] * warped + p[7]
        residual = (ref - model).ravel()
        rms = float(np.sqrt(np.mean(residual**2)))

        A = np.column_stack([
            gx_f * xc_f, gx_f * yc_f, gx_f,
            gy_f * xc_f, gy_f * yc_f, gy_f,
            warped.ravel(), np.ones(h * w),
        ])
        N = A.T @ A
        rhs = A.T @ residual
        try:
            delta = np.linalg.solve(N + damping * np.trace(N) / 8.0 * np.eye(8), rhs)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(delta).all():
            return None

        p = p + delta
        dof = max(h * w - 8, 1)
        sigma0_sq = float(residual @ residual / dof)

        if max(abs(delta[2]), abs(delta[5])) < tol_px:
            converged = True
            break

        if abs(p[0] - 1) > 0.5 or abs(p[4] - 1) > 0.5 or max(abs(p[2]), abs(p[5])) > w / 3.0:
            return None            # diverged out of the patch

    return LSMResult(
        dx=float(p[2]), dy=float(p[5]),
        affine=np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]]]),
        normal_matrix=N, sigma0_sq=sigma0_sq, n_iter=it, converged=converged,
        rms_residual=rms,
    )
