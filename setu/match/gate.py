"""S3 - the agreement gate. Novelty N2.

Two independent tracks run over the same pair: a pretrained cross-modality dense
network, and a handcrafted phase-congruency and maximum-index-map matcher. A
correspondence survives only if both tracks land within tau pixels of each other, or if
a single track's correlation peak is sharp enough to stand on its own.

The point is not to get more matches. It is to change the question from "how many
matches did you get" to "how many matches would you stake a landing site on". The two
tracks fail in uncorrelated ways - the network hallucinates on repetitive crater fields,
the handcrafted track drops out on low-texture mare - so their intersection is far
cleaner than either alone, and the agreement rate is itself a diagnostic worth reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
from scipy.spatial import cKDTree

from setu.illum.structural import cfog
from setu.match.base import MatchSet

Confidence = str  # "high" | "medium"


@dataclass
class GateResult:
    """Surviving correspondences plus the counts the QA report needs."""

    matches: MatchSet
    confidence: np.ndarray               # per-point "high" or "medium"
    origin: np.ndarray                   # per-point "agreed", "track_a_only", "track_b_only"
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_agreed(self) -> int:
        return int((self.origin == "agreed").sum())

    def summary(self) -> dict[str, Any]:
        return {
            "n_accepted": len(self.matches),
            "n_agreed": self.n_agreed,
            "n_track_a_only": int((self.origin == "track_a_only").sum()),
            "n_track_b_only": int((self.origin == "track_b_only").sum()),
            "n_high_confidence": int((self.confidence == "high").sum()),
            "n_medium_confidence": int((self.confidence == "medium").sum()),
            **self.stats,
        }


def _median_spacing(pts: np.ndarray, sample: int = 400) -> float:
    """Median nearest-neighbour distance of a point set, its effective grid spacing."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return 1.0
    if len(p) > sample:
        p = p[np.random.default_rng(0).choice(len(p), sample, replace=False)]
    d, _ = cKDTree(p).query(p, k=2)
    return float(np.median(d[:, 1]))


def correlation_sharpness(
    src_repr: np.ndarray,
    ref_repr: np.ndarray,
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    patch: int = 32,
    window: int = 24,
    metric: str = "cfog",
) -> tuple[np.ndarray, np.ndarray]:
    """Correlation peak height and peak-to-second-peak ratio for each correspondence.

    This is the measurement the single-track branch of the gate rests on, so it is
    computed rather than approximated from a network's confidence score. A network's
    confidence says how sure the network is; the peak ratio says whether the image
    content is actually unambiguous, and on a field of near-identical craters those two
    are not the same thing.
    """
    src = np.asarray(src_repr, dtype=np.float32)
    ref = np.asarray(ref_repr, dtype=np.float32)
    if metric == "cfog":
        src_t = cfog(src) if src.ndim == 2 else src
        ref_t = cfog(ref) if ref.ndim == 2 else ref
    else:
        src_t = src[:, :, None] if src.ndim == 2 else src
        ref_t = ref[:, :, None] if ref.ndim == 2 else ref

    from setu.match.structural import _correlate_tensor, _peak_sharpness

    half_p, half_w = patch // 2, window // 2
    peaks = np.zeros(len(pts_src), dtype=np.float64)
    ratios = np.zeros(len(pts_src), dtype=np.float64)

    for i, ((xs, ys), (xr, yr)) in enumerate(zip(np.atleast_2d(pts_src), np.atleast_2d(pts_ref))):
        c0, r0 = int(round(xs)) - half_p, int(round(ys)) - half_p
        if c0 < 0 or r0 < 0 or r0 + patch > src_t.shape[0] or c0 + patch > src_t.shape[1]:
            continue
        sc0, sr0 = int(round(xr)) - half_p - half_w, int(round(yr)) - half_p - half_w
        if sc0 < 0 or sr0 < 0 or sr0 + patch + window > ref_t.shape[0] or sc0 + patch + window > ref_t.shape[1]:
            continue

        surface = _correlate_tensor(
            ref_t[sr0:sr0 + patch + window, sc0:sc0 + patch + window],
            src_t[r0:r0 + patch, c0:c0 + patch],
        )
        if surface.size == 0:
            continue
        pk = np.unravel_index(int(np.argmax(surface)), surface.shape)
        peaks[i] = float(surface[pk])
        ratios[i] = _peak_sharpness(surface, pk, peaks[i])

    return peaks, ratios


def agreement_gate(
    track_a: MatchSet,
    track_b: MatchSet,
    tau_px: float = 2.0,
    src_tol_px: float | None = None,
    sharpness_peak_ratio: float = 1.5,
    sharpness_peak_min: float = 0.4,
    src_repr: np.ndarray | None = None,
    ref_repr: np.ndarray | None = None,
    sharpness_fn: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    max_single_track: int = 1200,
) -> GateResult:
    """Accept a correspondence when both tracks agree, or when one is sharply peaked.

        accept if (A) both tracks produce a correspondence for the same source
                      location within tau px of each other        -> high
             or   (B) only one track fired, but its correlation peak passes
                      peak / second_peak > 1.5 AND peak > 0.4     -> medium
             else reject

    "The same source location" is judged within `src_tol_px`, defaulting to tau. Track A
    is dense and track B is sparse, so agreement is resolved from B's points outward: for
    each B correspondence, the nearest A source point is found, and the two agree if their
    *reference* predictions coincide. Going the other way would let one B point claim
    dozens of A neighbours and inflate the agreement count.
    """
    src_tol = float(src_tol_px if src_tol_px is not None else tau_px)
    stats: dict[str, Any] = {
        "n_track_a_in": len(track_a), "n_track_b_in": len(track_b),
        "tau_px": tau_px, "src_tol_px": src_tol,
        "sharpness_peak_ratio": sharpness_peak_ratio, "sharpness_peak_min": sharpness_peak_min,
    }

    if track_a.is_empty and track_b.is_empty:
        return GateResult(MatchSet.empty("agreed"), np.array([]), np.array([]),
                          {**stats, "reason": "both tracks returned no correspondences"})

    accepted_src: list[np.ndarray] = []
    accepted_ref: list[np.ndarray] = []
    accepted_conf: list[float] = []
    confidence: list[str] = []
    origin: list[str] = []

    used_a = np.zeros(len(track_a), dtype=bool)
    used_b = np.zeros(len(track_b), dtype=bool)

    # ---- branch A: both tracks agree
    #
    # Track A is semi-dense on a stride-8 grid; track B is sparse and lands wherever
    # phase congruency peaks. Demanding that the two produce a point at the *same pixel*
    # would almost never fire, and would test grid alignment rather than agreement. What
    # is compared instead is the displacement each track predicts at track B's location:
    # track A's local displacement field is interpolated there from its nearest
    # neighbours, and the tracks agree when the two predicted reference positions fall
    # within tau of each other.
    if not track_a.is_empty and not track_b.is_empty:
        tree_a = cKDTree(track_a.kpts_src)
        disp_a = track_a.kpts_ref - track_a.kpts_src
        k = int(min(4, len(track_a)))
        radius = max(src_tol, 2.0 * _median_spacing(track_a.kpts_src))
        stats["track_a_spacing_px"] = round(_median_spacing(track_a.kpts_src), 2)
        stats["agreement_radius_px"] = round(radius, 2)

        dists, idxs = tree_a.query(track_b.kpts_src, k=k, distance_upper_bound=radius)
        dists = np.atleast_2d(dists.T).T if k > 1 else dists[:, None]
        idxs = np.atleast_2d(idxs.T).T if k > 1 else idxs[:, None]

        for bi in range(len(track_b)):
            near = idxs[bi][np.isfinite(dists[bi]) & (idxs[bi] < len(track_a))]
            if near.size == 0:
                continue
            d = dists[bi][:near.size]
            # Inverse-distance weights, so a neighbour sitting on the point dominates.
            w = 1.0 / np.maximum(d, 1e-3)
            w = w / w.sum()
            pred_ref = track_b.kpts_src[bi] + (disp_a[near] * w[:, None]).sum(axis=0)

            disagreement = float(np.hypot(*(pred_ref - track_b.kpts_ref[bi])))
            if disagreement > tau_px:
                continue

            wa = float(np.average(track_a.conf[near], weights=w))
            wb = float(track_b.conf[bi])
            total = max(wa + wb, 1e-9)
            accepted_src.append(track_b.kpts_src[bi])
            accepted_ref.append((pred_ref * wa + track_b.kpts_ref[bi] * wb) / total)
            accepted_conf.append(float(np.clip(0.5 * (wa + wb) + 0.25, 0.0, 1.0)))
            confidence.append("high")
            origin.append("agreed")
            used_b[bi] = True
            # The track A points that voted are consumed by the agreement and are not
            # re-offered to the single-track branch.
            used_a[near] = True

    stats["n_agreed_pairs"] = len(accepted_src)
    stats["agreement_rate_b"] = float(used_b.mean()) if len(track_b) else 0.0

    # ---- branch B: single-track survivors that pass the sharpness test
    singles: list[tuple[MatchSet, np.ndarray, str]] = []
    if not track_a.is_empty and (~used_a).any():
        singles.append((track_a, ~used_a, "track_a_only"))
    if not track_b.is_empty and (~used_b).any():
        singles.append((track_b, ~used_b, "track_b_only"))

    n_tested = n_passed = 0
    for ms, mask, label in singles:
        cand = ms.filter(mask)
        if len(cand) > max_single_track:
            # Sharpness is a per-point correlation and is the expensive part of the gate.
            # The highest-confidence candidates are tested; the rest are dropped rather
            # than admitted untested, which keeps the gate's guarantee intact.
            stats[f"{label}_untested_dropped"] = len(cand) - max_single_track
            cand = cand.top(max_single_track)

        if sharpness_fn is not None:
            peaks, ratios = sharpness_fn(cand)
        elif "peak" in cand.meta and "sharpness" in cand.meta and len(cand.meta["peak"]) == len(cand):
            peaks = np.asarray(cand.meta["peak"], dtype=np.float64)
            ratios = np.asarray(cand.meta["sharpness"], dtype=np.float64)
        elif src_repr is not None and ref_repr is not None:
            peaks, ratios = correlation_sharpness(src_repr, ref_repr, cand.kpts_src, cand.kpts_ref)
        else:
            # No way to measure sharpness means no way to honour branch B. Rejecting is
            # the correct behaviour; admitting these on confidence alone would quietly
            # turn the gate off.
            stats[f"{label}_unmeasurable"] = len(cand)
            continue

        n_tested += len(cand)
        ok = (ratios > sharpness_peak_ratio) & (peaks > sharpness_peak_min)
        n_passed += int(ok.sum())
        for i in np.nonzero(ok)[0]:
            accepted_src.append(cand.kpts_src[i])
            accepted_ref.append(cand.kpts_ref[i])
            accepted_conf.append(float(np.clip(0.6 * cand.conf[i] + 0.2 * min(ratios[i] / 3.0, 1.0), 0.0, 1.0)))
            confidence.append("medium")
            origin.append(label)

    stats["n_single_track_tested"] = n_tested
    stats["n_single_track_passed"] = n_passed
    stats["single_track_pass_rate"] = float(n_passed / n_tested) if n_tested else 0.0

    if not accepted_src:
        return GateResult(MatchSet.empty("agreed"), np.array([]), np.array([]),
                          {**stats, "reason": "no correspondence passed either branch of the gate"})

    matches = MatchSet(
        np.vstack(accepted_src), np.vstack(accepted_ref), np.asarray(accepted_conf),
        track="agreed", meta={"gate": stats},
    )
    return GateResult(matches, np.asarray(confidence), np.asarray(origin), stats)


def false_match_rate(
    matches: MatchSet,
    H_true: np.ndarray,
    tol_px: float = 3.0,
) -> dict[str, float]:
    """False-match rate against exact truth - the number N2 is measured by.

    Only computable on the synthetic benchmark, which is the reason the benchmark exists:
    on real data there is no way to count false matches, only to notice that the fit got
    worse.
    """
    from setu.bench.generate import apply_h

    if matches.is_empty:
        return {"false_match_rate": float("nan"), "n": 0, "n_correct": 0}
    err = np.hypot(*(matches.kpts_ref - apply_h(H_true, matches.kpts_src)).T)
    correct = int((err <= tol_px).sum())
    return {
        "false_match_rate": float(1.0 - correct / len(matches)),
        "precision": float(correct / len(matches)),
        "n": len(matches),
        "n_correct": correct,
        "median_error_px": float(np.median(err)),
    }
