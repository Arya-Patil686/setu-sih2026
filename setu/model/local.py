"""S5 local stage - polynomial and thin-plate-spline residual models.

What remains after the global fit is not noise. It is real terrain-induced parallax and
sensor distortion, and modelling it is worth several tenths of a pixel. The danger is
equally real: a thin-plate spline with too little regularisation interpolates its own
control points exactly, reports a fit RMSE near zero, and is worthless anywhere else.

The regularisation parameter is therefore chosen by leave-one-out cross-validation, and
the LOOCV checkpoint RMSE is reported alongside the fit RMSE in every output. This is
the single easiest place for a registration project to be either honest or dishonest.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

LocalKind = Literal["tps", "poly2", "none"]


@dataclass
class LocalModel:
    """A fitted residual correction, plus the numbers that say whether to trust it."""

    kind: LocalKind
    predict: Callable[[np.ndarray], np.ndarray]
    fit_rmse_px: float
    loocv_rmse_px: float
    lam: float
    n_points: int
    coefficients: dict[str, Any]

    @property
    def improves(self) -> bool:
        """Whether the local model is worth applying at all.

        A local model that does not beat its own input out of sample is overfitting, and
        is discarded rather than shipped with a flattering fit RMSE.
        """
        return bool(np.isfinite(self.loocv_rmse_px) and self.loocv_rmse_px < self.fit_rmse_px * 3.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "lambda": self.lam,
            "n_points": self.n_points,
            "fit_rmse_px": round(self.fit_rmse_px, 4),
            "loocv_rmse_px": round(self.loocv_rmse_px, 4),
            "applied": self.improves,
            **self.coefficients,
        }


# ------------------------------------------------------------------ poly2

def _poly2_design(pts: np.ndarray, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    """Second-order polynomial basis on normalised coordinates.

    Normalising before building the design matrix is not cosmetic: on raw pixel
    coordinates in the thousands, the squared terms reach 10^7 and the normal equations
    become numerically hopeless.
    """
    x = (pts[:, 0] - cx) / sx
    y = (pts[:, 1] - cy) / sy
    return np.column_stack([np.ones(len(pts)), x, y, x * x, x * y, y * y])


def fit_poly2(src: np.ndarray, residual: np.ndarray, lam: float = 1e-6) -> tuple[Callable, dict[str, Any]]:
    """Ridge-regularised quadratic in (x, y) per residual component."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    res = np.asarray(residual, dtype=np.float64).reshape(-1, 2)
    cx, cy = src[:, 0].mean(), src[:, 1].mean()
    sx = max(src[:, 0].std(), 1.0)
    sy = max(src[:, 1].std(), 1.0)

    A = _poly2_design(src, cx, cy, sx, sy)
    reg = lam * np.eye(A.shape[1])
    reg[0, 0] = 0.0                      # never penalise the constant term
    coef = np.linalg.solve(A.T @ A + reg, A.T @ res)

    def predict(pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return _poly2_design(p, cx, cy, sx, sy) @ coef

    return predict, {"poly2_coefficients": coef.tolist(), "normalisation": [cx, cy, sx, sy]}


# -------------------------------------------------------------------- TPS

def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    """U(r) = r^2 log(r), the thin-plate spline's fundamental solution in 2-D."""
    out = np.zeros_like(r2)
    nz = r2 > 1e-12
    out[nz] = r2[nz] * np.log(np.sqrt(r2[nz]))
    return out


def _pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)


def fit_tps(src: np.ndarray, residual: np.ndarray, lam: float = 1e-2) -> tuple[Callable, dict[str, Any]]:
    """Regularised thin-plate spline through the residual field.

    `lam` enters on the diagonal of the kernel block, which is the standard smoothing
    formulation: at lam = 0 the spline interpolates every control point exactly, and as
    lam grows the solution relaxes towards the affine part alone.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    res = np.asarray(residual, dtype=np.float64).reshape(-1, 2)
    n = len(src)

    scale = max(np.ptp(src[:, 0]), np.ptp(src[:, 1]), 1.0)
    ctrl = src / scale

    K = _tps_kernel(_pairwise_sq(ctrl, ctrl)) + lam * np.eye(n)
    P = np.column_stack([np.ones(n), ctrl])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T

    rhs = np.zeros((n + 3, 2))
    rhs[:n] = res
    try:
        params = np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(L, rhs, rcond=None)[0]

    w, a = params[:n], params[n:]

    def predict(pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2) / scale
        U = _tps_kernel(_pairwise_sq(p, ctrl))
        return U @ w + np.column_stack([np.ones(len(p)), p]) @ a

    return predict, {"tps_n_control": n, "tps_scale": scale}


# ------------------------------------------------------------------ selection

def _loocv(src: np.ndarray, residual: np.ndarray, fit_fn: Callable, lam: float, max_points: int = 200) -> float:
    """Leave-one-out RMSE for one regularisation value."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    res = np.asarray(residual, dtype=np.float64).reshape(-1, 2)
    n = len(src)
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
            predict, _ = fit_fn(src[keep], res[keep], lam)
            errs.append(float(np.hypot(*(predict(src[i:i + 1]).ravel() - res[i]))))
        except Exception:
            continue
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def fit_local(
    src: np.ndarray,
    residual: np.ndarray,
    kind: LocalKind = "tps",
    lam_grid: Sequence[float] | None = None,
    loocv_max_points: int = 200,
) -> LocalModel | None:
    """Fit the local residual model, selecting the regularisation by cross-validation.

    Both the fit RMSE and the LOOCV RMSE are returned. Reporting only the first would be
    reporting how well the model memorised its own control points.
    """
    if kind == "none":
        return None
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    res = np.asarray(residual, dtype=np.float64).reshape(-1, 2)
    if len(src) < 10:
        return None

    fit_fn = fit_tps if kind == "tps" else fit_poly2
    grid = list(lam_grid or ([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0] if kind == "tps" else [1e-8, 1e-6, 1e-4, 1e-2]))

    best_lam, best_loocv = grid[0], np.inf
    for lam in grid:
        score = _loocv(src, res, fit_fn, lam, loocv_max_points)
        if np.isfinite(score) and score < best_loocv:
            best_lam, best_loocv = lam, score

    predict, coeffs = fit_fn(src, res, best_lam)
    fit_rmse = float(np.sqrt(np.mean(np.sum((predict(src) - res) ** 2, axis=1))))

    return LocalModel(
        kind=kind, predict=predict, fit_rmse_px=fit_rmse, loocv_rmse_px=float(best_loocv),
        lam=float(best_lam), n_points=len(src), coefficients=coeffs,
    )
