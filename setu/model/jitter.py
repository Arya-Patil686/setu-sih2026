"""S5 - per-line attitude jitter for pushbroom sensors.

OHRC and TMC-2 are line-scan instruments: each image row is a separate exposure taken at
a slightly different spacecraft attitude. A global affine cannot represent that, because
the distortion is a function of row alone and varies along the strip. Attitude jitter of
the order of the image GSD has been documented for OHRC stereo products, so this is a
real signal and not a modelling flourish.

Fitting a smooth spline to dx(row) and dy(row) and reporting its amplitude and dominant
frequency is genuinely mission-relevant output: it is a measurement of the spacecraft's
pointing stability, extracted from the registration as a by-product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.interpolate import LSQUnivariateSpline


@dataclass
class JitterModel:
    """Smooth dx(row), dy(row) along a pushbroom strip."""

    rows: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    amplitude_px: float
    dominant_period_rows: float
    n_knots: int
    n_points: int
    rms_reduction: float

    def predict(self, rows: np.ndarray) -> np.ndarray:
        """Interpolated (dx, dy) for arbitrary rows, clamped outside the fitted range."""
        r = np.clip(np.asarray(rows, dtype=np.float64), self.rows[0], self.rows[-1])
        return np.column_stack([np.interp(r, self.rows, self.dx), np.interp(r, self.rows, self.dy)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "amplitude_px": round(float(self.amplitude_px), 4),
            "dominant_period_rows": round(float(self.dominant_period_rows), 2),
            "n_knots": self.n_knots,
            "n_points": self.n_points,
            "rms_reduction_px": round(float(self.rms_reduction), 4),
            "rows": self.rows.tolist(),
            "dx": np.round(self.dx, 5).tolist(),
            "dy": np.round(self.dy, 5).tolist(),
        }


def fit_jitter(
    src_rows: Sequence[float],
    residual: np.ndarray,
    n_knots: int = 12,
    n_samples: int = 200,
    min_points: int = 40,
) -> JitterModel | None:
    """Fit a least-squares spline to the row-dependent part of the residual.

    Interior knots are placed at quantiles of the observed rows rather than uniformly,
    so a strip whose tie points are unevenly distributed along-track does not end up
    with unconstrained knots in its sparse regions.
    """
    rows = np.asarray(src_rows, dtype=np.float64).ravel()
    res = np.asarray(residual, dtype=np.float64).reshape(-1, 2)
    if len(rows) < min_points:
        return None

    order = np.argsort(rows)
    rows, res = rows[order], res[order]
    # A spline cannot be fitted through duplicate abscissae; averaging repeated rows is
    # the right reduction because they are genuinely repeat observations of one attitude.
    uniq, inverse = np.unique(rows, return_inverse=True)
    if len(uniq) < min_points:
        return None
    res = np.column_stack([
        np.bincount(inverse, weights=res[:, 0]) / np.bincount(inverse),
        np.bincount(inverse, weights=res[:, 1]) / np.bincount(inverse),
    ])
    rows = uniq

    k = min(n_knots, max(2, len(rows) // 8))
    interior = np.quantile(rows, np.linspace(0, 1, k + 2)[1:-1])
    interior = np.unique(interior)
    interior = interior[(interior > rows[0]) & (interior < rows[-1])]
    if interior.size == 0:
        return None

    try:
        sx = LSQUnivariateSpline(rows, res[:, 0], interior, k=3)
        sy = LSQUnivariateSpline(rows, res[:, 1], interior, k=3)
    except Exception:
        return None

    sample_rows = np.linspace(rows[0], rows[-1], n_samples)
    dx, dy = sx(sample_rows), sy(sample_rows)

    before = float(np.sqrt(np.mean(res[:, 0] ** 2 + res[:, 1] ** 2)))
    after = float(np.sqrt(np.mean((res[:, 0] - sx(rows)) ** 2 + (res[:, 1] - sy(rows)) ** 2)))

    return JitterModel(
        rows=sample_rows, dx=dx, dy=dy,
        amplitude_px=float(np.sqrt(np.ptp(dx) ** 2 + np.ptp(dy) ** 2) / 2.0),
        dominant_period_rows=_dominant_period(sample_rows, dx, dy),
        n_knots=int(interior.size), n_points=int(len(rows)),
        rms_reduction=before - after,
    )


def _dominant_period(rows: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> float:
    """Period of the strongest non-DC component of the jitter signature, in rows.

    Reported because a periodic signature points at a specific mechanical source - a
    reaction wheel, a cooler, a solar-array drive - whereas a broadband one does not.
    """
    signal = np.hypot(dx - dx.mean(), dy - dy.mean())
    if signal.size < 8 or signal.std() < 1e-9:
        return float("nan")
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    if spectrum.size < 2:
        return float("nan")
    peak = int(np.argmax(spectrum[1:])) + 1
    span = float(rows[-1] - rows[0])
    return float(span / peak) if peak > 0 else float("nan")
