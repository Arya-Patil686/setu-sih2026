"""S4 - the refinement stage, end to end.

Every accepted correspondence goes through upsampled phase correlation and then
least-squares matching, both on the structural representations, and comes out carrying a
covariance.

One thing has to happen before any of that is valid. Phase correlation measures a pure
translation. Two 64 x 64 patches taken from images that differ by even a modest rotation
are *not* related by a translation: at 12 degrees, a patch corner moves nearly seven
pixels, and the correlation peak that comes back is not a refinement of anything. So a
provisional global model is fitted to the gated matches first, and each source patch is
resampled through the local linearisation of that model before it is correlated. The two
patches then really do differ by a small translation, which is what both refiners assume.

Points whose uncertainty exceeds the configured ceiling, or whose refinement moved them
further than the prior says is plausible, are dropped here rather than being passed on
with a plausible-looking position and no warning attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from setu.match.base import MatchSet
from setu.refine.covariance import (
    apply_variance_scale,
    covariance_from_lsm,
    covariance_from_paraboloid,
    covariance_to_fields,
    variance_component_scale,
)
from setu.refine.lsm import least_squares_match
from setu.refine.phasecorr import correlation_surface, subpixel_patch
from setu.types import TiePoint


@dataclass
class RefineReport:
    """What happened to the correspondences that entered S4."""

    n_in: int = 0
    n_phase_ok: int = 0
    n_lsm_converged: int = 0
    n_rejected_sigma: int = 0
    n_rejected_patch: int = 0
    n_rejected_drift: int = 0
    n_out: int = 0
    variance_factor: float = 1.0
    prior_model: str = "none"
    median_shift_px: float = float("nan")
    method_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_in": self.n_in,
            "n_phase_correlation_ok": self.n_phase_ok,
            "n_lsm_converged": self.n_lsm_converged,
            "n_rejected_high_sigma": self.n_rejected_sigma,
            "n_rejected_patch_out_of_bounds": self.n_rejected_patch,
            "n_rejected_drift": self.n_rejected_drift,
            "n_out": self.n_out,
            "sigma_variance_factor": round(self.variance_factor, 4),
            "prior_model": self.prior_model,
            "median_refinement_shift_px": round(self.median_shift_px, 4)
            if np.isfinite(self.median_shift_px) else None,
            "covariance_method_counts": self.method_counts,
        }


def provisional_model(matches: MatchSet) -> tuple[np.ndarray | None, np.ndarray, str]:
    """Robust affine prior over the gated matches, with its inlier mask.

    Full affine rather than similarity: the residual after pre-alignment genuinely
    contains shear and differential scale, and a four-parameter prior would leave that in
    the residual and inflate every uncertainty estimated from it.
    """
    if len(matches) < 6:
        return None, np.zeros(len(matches), bool), "none"

    src = matches.kpts_src.astype(np.float32).reshape(-1, 1, 2)
    ref = matches.kpts_ref.astype(np.float32).reshape(-1, 1, 2)
    try:
        M, mask = cv2.estimateAffine2D(src, ref, method=cv2.USAC_MAGSAC,
                                       ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999)
    except cv2.error:
        M, mask = None, None
    if M is None:
        return None, np.zeros(len(matches), bool), "none"

    H = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64)
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(matches), bool)
    return H, inliers, "affine"


def local_jacobian(H: np.ndarray, pt: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """2 x 2 linearisation of H at a point - the local rotation, scale and shear."""
    from setu.bench.generate import apply_h

    p = np.asarray(pt, dtype=np.float64).reshape(1, 2)
    base = apply_h(H, p)[0]
    dx = apply_h(H, p + np.array([[eps, 0.0]]))[0] - base
    dy = apply_h(H, p + np.array([[0.0, eps]]))[0] - base
    return np.column_stack([dx, dy]) / eps


def rectified_patch(
    image: np.ndarray,
    centre: np.ndarray,
    jacobian: np.ndarray,
    size: int,
) -> np.ndarray | None:
    """Source patch resampled into the reference's local orientation and scale.

    The patch is sampled so that its axes line up with the reference frame's, which is
    what makes the subsequent translation-only correlation meaningful.
    """
    try:
        J_inv = np.linalg.inv(np.asarray(jacobian, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None

    half = size / 2.0
    # Output pixel (u, v) samples source pixel centre + J^-1 . (u - half, v - half), so
    # output index `half` lands exactly on `centre`, matching `subpixel_patch`.
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = J_inv
    M[:, 2] = np.asarray(centre, dtype=np.float64) - J_inv @ np.array([half, half])

    h, w = image.shape[:2]
    corners = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float64)
    mapped = (corners @ M[:, :2].T) + M[:, 2]
    if mapped[:, 0].min() < 0 or mapped[:, 1].min() < 0 or mapped[:, 0].max() >= w or mapped[:, 1].max() >= h:
        return None

    patch = cv2.warpAffine(
        np.ascontiguousarray(image, dtype=np.float32), M, (size, size),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT101,
    )
    return patch if np.isfinite(patch).all() else None


def _phase_shift(a: np.ndarray, b: np.ndarray, upsample_factor: int, max_shift_px: float):
    """Translation from a to b by upsampled phase correlation, bounded by max_shift_px."""
    from skimage.registration import phase_cross_correlation

    if a.std() < 1e-8 or b.std() < 1e-8:
        return None
    size = a.shape[0]
    win = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)
    aw = (a - a.mean()) * win
    bw = (b - b.mean()) * win
    try:
        shift, error, _ = phase_cross_correlation(bw, aw, upsample_factor=upsample_factor,
                                                  normalization="phase")
    except Exception:
        return None
    dy, dx = float(shift[0]), float(shift[1])
    if not np.isfinite(dx) or not np.isfinite(dy) or np.hypot(dx, dy) > max_shift_px:
        return None
    return dx, dy, float(error)


def refine_matches(
    matches: MatchSet,
    src_repr: np.ndarray,
    ref_repr: np.ndarray,
    patch: int = 64,
    upsample_factor: int = 50,
    use_lsm: bool = True,
    lsm_max_iter: int = 12,
    lsm_tol_px: float = 1e-3,
    max_sigma_px: float = 0.5,
    covariance_method: str = "auto",
    max_shift_px: float = 4.0,
    confidence: np.ndarray | None = None,
    origin: np.ndarray | None = None,
) -> tuple[list[TiePoint], RefineReport]:
    """Refine every correspondence to sub-pixel and attach its covariance."""
    report = RefineReport(n_in=len(matches), method_counts={"lsm": 0, "paraboloid": 0, "none": 0})
    if matches.is_empty:
        return [], report

    H_prior, prior_inliers, prior_kind = provisional_model(matches)
    report.prior_model = prior_kind

    raw: list[dict[str, Any]] = []
    shifts: list[float] = []

    for i in range(len(matches)):
        p_src = matches.kpts_src[i]
        p_ref = matches.kpts_ref[i]

        J = local_jacobian(H_prior, p_src) if H_prior is not None else np.eye(2)
        a = rectified_patch(src_repr, p_src, J, patch)
        b = subpixel_patch(ref_repr, p_ref[0], p_ref[1], patch)
        if a is None or b is None:
            report.n_rejected_patch += 1
            continue

        shifted = _phase_shift(a, b, upsample_factor, max_shift_px)
        if shifted is None:
            report.n_rejected_drift += 1
            continue
        dx, dy, pc_error = shifted
        report.n_phase_ok += 1
        p_ref_refined = np.array([p_ref[0] + dx, p_ref[1] + dy])

        b2 = subpixel_patch(ref_repr, p_ref_refined[0], p_ref_refined[1], patch)
        cov, method = None, "none"

        if use_lsm and b2 is not None:
            res = least_squares_match(a, b2, max_iter=lsm_max_iter, tol_px=lsm_tol_px)
            if res is not None and res.converged and abs(res.dx) < max_shift_px and abs(res.dy) < max_shift_px:
                report.n_lsm_converged += 1
                p_ref_refined = p_ref_refined + np.array([res.dx, res.dy])
                cov = covariance_from_lsm(res.normal_matrix, _lsm_residual(a, b2, res.affine))
                method = "lsm"

        if (cov is None or not np.isfinite(cov).all()) and b2 is not None:
            surface = correlation_surface(b2, a)
            pk = np.unravel_index(int(np.argmax(surface)), surface.shape)
            _, cov = covariance_from_paraboloid(surface, pk)
            method = "paraboloid" if cov is not None and np.isfinite(cov).all() else "none"

        total_shift = float(np.hypot(*(p_ref_refined - p_ref)))
        shifts.append(total_shift)
        raw.append({
            "idx": i, "src": p_src, "ref": p_ref_refined, "cov": cov, "method": method,
            "conf": float(matches.conf[i]), "pc_error": pc_error, "shift": total_shift,
        })

    if not raw:
        return [], report

    report.median_shift_px = float(np.median(shifts)) if shifts else float("nan")

    # The variance factor comes from the residuals of the *prior* model on the refined
    # points, which is an honest external check: the prior was fitted before refinement,
    # so it cannot have absorbed the refinement's own error.
    residuals = _residuals_against(H_prior, raw) if H_prior is not None else np.full(len(raw), np.nan)
    sigmas = np.array([
        np.sqrt(np.trace(r["cov"])) if r["cov"] is not None and np.isfinite(r["cov"]).all() else np.nan
        for r in raw
    ])
    report.variance_factor = variance_component_scale(sigmas, residuals)

    tiepoints: list[TiePoint] = []
    for tid, r in enumerate(raw):
        cov = apply_variance_scale(r["cov"], report.variance_factor) if r["cov"] is not None else None
        sx, sy, sxy = covariance_to_fields(cov) if cov is not None else (np.nan, np.nan, 0.0)
        report.method_counts[r["method"]] = report.method_counts.get(r["method"], 0) + 1

        if np.isfinite(sx) and np.isfinite(sy) and np.hypot(sx, sy) > max_sigma_px:
            report.n_rejected_sigma += 1
            continue

        i = r["idx"]
        tiepoints.append(TiePoint(
            tid=tid,
            src_sample=float(r["src"][0]), src_line=float(r["src"][1]),
            ref_sample=float(r["ref"][0]), ref_line=float(r["ref"][1]),
            conf=r["conf"],
            track=str(origin[i]) if origin is not None and i < len(origin) else matches.track,
            sigma_x=sx, sigma_y=sy, sigma_xy=sxy,
        ))

    report.n_out = len(tiepoints)
    return tiepoints, report


def _residuals_against(H: np.ndarray, raw: list[dict[str, Any]]) -> np.ndarray:
    from setu.bench.generate import apply_h

    src = np.array([r["src"] for r in raw], dtype=np.float64)
    ref = np.array([r["ref"] for r in raw], dtype=np.float64)
    return np.hypot(*(apply_h(H, src) - ref).T)


def _lsm_residual(src_patch: np.ndarray, ref_patch: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Residual image at LSM convergence, needed for the covariance's variance term."""
    from scipy.ndimage import map_coordinates

    h, w = ref_patch.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sx = affine[0, 0] * (xx - cx) + affine[0, 1] * (yy - cy) + affine[0, 2] + cx
    sy = affine[1, 0] * (xx - cx) + affine[1, 1] * (yy - cy) + affine[1, 2] + cy
    warped = map_coordinates(
        np.asarray(src_patch, dtype=np.float64), [sy.ravel(), sx.ravel()], order=3, mode="reflect"
    ).reshape(h, w)
    gain = ref_patch.std() / max(warped.std(), 1e-8)
    return np.asarray(ref_patch, dtype=np.float64) - (gain * warped + (ref_patch.mean() - gain * warped.mean()))
