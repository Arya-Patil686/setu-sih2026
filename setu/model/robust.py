"""S5 global stage - robust estimation with MAGSAC++.

Two things distinguish this from a default RANSAC call. The model is chosen by scene
geometry rather than fixed: affine for small, well-conditioned footprints, projective
once the emission-angle difference makes a planar homography genuinely necessary, and
similarity as the conservative fallback when there are too few inliers to support more
parameters. And the inlier threshold is *derived from the per-point covariances of S4*
rather than being a magic constant - a run whose points are genuinely more precise gets
a tighter threshold, automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import cv2
import numpy as np

from setu.types import TiePoint

ModelKind = Literal["affine", "homography", "similarity"]


@dataclass
class GlobalFit:
    """The estimated global transform and everything needed to justify it."""

    H: np.ndarray
    kind: ModelKind
    inliers: np.ndarray            # boolean mask over the input points
    threshold_px: float
    n_input: int
    residuals: np.ndarray

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())

    @property
    def inlier_ratio(self) -> float:
        return float(self.n_inliers / self.n_input) if self.n_input else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "matrix": self.H.tolist(),
            "threshold_px": round(float(self.threshold_px), 4),
            "n_input": self.n_input,
            "n_inliers": self.n_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "inlier_rmse_px": round(float(np.sqrt(np.mean(self.residuals[self.inliers] ** 2))), 4)
            if self.n_inliers else float("nan"),
        }


def adaptive_threshold(
    tiepoints: Sequence[TiePoint],
    scale: float = 3.0,
    floor_px: float = 0.75,
    ceiling_px: float = 8.0,
) -> float:
    """Inlier threshold from the points' own uncertainties: 3 x median(sqrt(trace Sigma)).

    The floor and ceiling are guards, not the operating point. Without a floor, a run
    whose covariances came out implausibly small would reject every true match; without
    a ceiling, one bad run would accept everything.
    """
    sig = np.array([t.sigma_rms for t in tiepoints], dtype=np.float64)
    sig = sig[np.isfinite(sig) & (sig > 0)]
    if sig.size < 5:
        return float(np.clip(2.0, floor_px, ceiling_px))
    return float(np.clip(scale * np.median(sig), floor_px, ceiling_px))


def choose_model(
    n_points: int,
    emission_diff_deg: float,
    preference: str = "auto",
    homography_emission_deg: float = 10.0,
    similarity_min_inliers: int = 30,
) -> ModelKind:
    """Pick the global model from scene geometry and the number of points available."""
    if preference in ("affine", "homography", "similarity"):
        return preference                                    # type: ignore[return-value]
    if n_points < similarity_min_inliers:
        return "similarity"
    if emission_diff_deg > homography_emission_deg:
        return "homography"
    return "affine"


def _to_h(matrix: np.ndarray, kind: ModelKind) -> np.ndarray:
    if kind == "homography":
        return np.asarray(matrix, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    return np.vstack([m, [0.0, 0.0, 1.0]])


def fit_global(
    tiepoints: Sequence[TiePoint],
    kind: ModelKind = "affine",
    threshold_px: float = 2.0,
    confidence: float = 0.9999,
    max_iters: int = 10000,
) -> GlobalFit | None:
    """Robustly estimate the global transform with MAGSAC++.

    MAGSAC++ marginalises over the noise scale instead of committing to one threshold,
    which is why it is specified here: the threshold passed in is a soft upper bound
    rather than a hard decision boundary, and that is exactly the right behaviour when
    the point uncertainties vary by an order of magnitude across the image.
    """
    from setu.bench.generate import apply_h

    pts = list(tiepoints)
    if len(pts) < 4:
        return None

    src = np.array([[t.src_sample, t.src_line] for t in pts], dtype=np.float32).reshape(-1, 1, 2)
    ref = np.array([[t.ref_sample, t.ref_line] for t in pts], dtype=np.float32).reshape(-1, 1, 2)

    try:
        if kind == "homography":
            M, mask = cv2.findHomography(src, ref, cv2.USAC_MAGSAC, threshold_px,
                                         maxIters=max_iters, confidence=confidence)
        elif kind == "affine":
            M, mask = cv2.estimateAffine2D(src, ref, method=cv2.USAC_MAGSAC,
                                           ransacReprojThreshold=threshold_px,
                                           maxIters=max_iters, confidence=confidence)
        else:
            M, mask = cv2.estimateAffinePartial2D(src, ref, method=cv2.USAC_MAGSAC,
                                                  ransacReprojThreshold=threshold_px,
                                                  maxIters=max_iters, confidence=confidence)
    except cv2.error:
        return None

    if M is None or mask is None:
        return None

    H = _to_h(M, kind)
    inliers = mask.ravel().astype(bool)
    src_flat = src.reshape(-1, 2).astype(np.float64)
    residuals = np.hypot(*(apply_h(H, src_flat) - ref.reshape(-1, 2)).T)

    return GlobalFit(H=H, kind=kind, inliers=inliers, threshold_px=threshold_px,
                     n_input=len(pts), residuals=residuals)


def fit_global_auto(
    tiepoints: Sequence[TiePoint],
    preference: str = "auto",
    emission_diff_deg: float = 0.0,
    threshold_scale: float = 3.0,
    floor_px: float = 0.75,
    ceiling_px: float = 8.0,
    homography_emission_deg: float = 10.0,
    similarity_min_inliers: int = 30,
    confidence: float = 0.9999,
    max_iters: int = 10000,
) -> GlobalFit | None:
    """Choose the model, derive the threshold from the covariances, and fit.

    If the chosen model yields too few inliers, the fit is retried with the
    conservative similarity model. A projective fit on 20 noisy points is worse than no
    projective fit at all: it will happily bend the scene to accommodate the outliers.
    """
    thr = adaptive_threshold(tiepoints, threshold_scale, floor_px, ceiling_px)
    kind = choose_model(len(tiepoints), emission_diff_deg, preference,
                        homography_emission_deg, similarity_min_inliers)

    fit = fit_global(tiepoints, kind, thr, confidence, max_iters)
    if fit is not None and fit.n_inliers >= max(8, similarity_min_inliers // 3):
        return fit

    if kind != "similarity":
        fallback = fit_global(tiepoints, "similarity", thr, confidence, max_iters)
        if fallback is not None:
            return fallback
    return fit


def mark_inliers(tiepoints: Sequence[TiePoint], fit: GlobalFit) -> None:
    """Write residuals and inlier flags back onto the tie points, in place.

    The fit's own boolean mask is authoritative for the points it was estimated from.
    Re-seeded points arrive after the fit and have no entry in that mask, so their status
    is decided by testing their residual against the same threshold - which is the
    consistent rule, and keeps a re-seeded point from being trusted purely because it was
    added late.
    """
    from setu.bench.generate import apply_h

    pts = list(tiepoints)
    if not pts:
        return

    src = np.array([[t.src_sample, t.src_line] for t in pts], dtype=np.float64)
    pred = apply_h(fit.H, src)
    from_fit = len(pts) == len(fit.inliers)

    for i, t in enumerate(pts):
        t.residual_x = float(pred[i, 0] - t.ref_sample)
        t.residual_y = float(pred[i, 1] - t.ref_line)
        if from_fit:
            t.inlier = bool(fit.inliers[i])
        else:
            t.inlier = bool(t.residual_norm <= fit.threshold_px)
