"""S6 step 4 - re-seeding empty cells.

This is the step that actually delivers uniformity, and it is the S6 -> S3 feedback edge
of the architecture. For every valid cell that ended up with no surviving inlier, track B
is re-run *inside that cell only*, under three changes that make success far more likely
than it was on the first pass:

  1. the global transform from S5 supplies the initial guess, so the search starts in the
     right place instead of anywhere in the reference;
  2. the search window shrinks to a few pixels, because the answer is now known to be
     nearby - which also means a repeated crater elsewhere can no longer be a candidate;
  3. the acceptance threshold is relaxed, because a weak match constrained to within a
     few pixels of a known answer is far safer than a strong one found by open search.

Two passes maximum. A cell that fails twice under those conditions has no matchable
texture, and forcing a point into it would be manufacturing a correspondence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from setu.bench.generate import apply_h
from setu.match.base import MatchSet
from setu.match.structural import template_refine
from setu.types import TiePoint
from setu.uniform.lattice import Lattice


@dataclass
class ReseedReport:
    passes: int = 0
    cells_targeted: int = 0
    cells_filled: int = 0
    points_added: int = 0
    coverage_before: float = 0.0
    coverage_after: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passes": self.passes,
            "cells_targeted": self.cells_targeted,
            "cells_filled": self.cells_filled,
            "points_added": self.points_added,
            "coverage_before": round(self.coverage_before, 4),
            "coverage_after": round(self.coverage_after, 4),
        }


def reseed_cell(
    src_repr: np.ndarray,
    ref_repr: np.ndarray,
    lattice: Lattice,
    cell_id: int,
    H_global: np.ndarray,
    window_px: int = 5,
    n_candidates: int = 6,
    min_peak: float = 0.25,
    patch: int = 32,
) -> MatchSet:
    """Search one cell for correspondences, using the global transform as the prior.

    Candidate locations come from phase congruency inside the cell; each is projected
    through the global model and refined by a template search over a window of only a
    few pixels.
    """
    from setu.illum.structural import phase_congruency
    from setu.match.structural import detect_pc_keypoints

    r0, c0, r1, c1 = lattice.cell_bounds(cell_id)
    pad = patch // 2 + window_px + 4
    sr0, sc0 = max(0, r0 - pad), max(0, c0 - pad)
    sr1, sc1 = min(src_repr.shape[0], r1 + pad), min(src_repr.shape[1], c1 + pad)
    if sr1 - sr0 < patch + 4 or sc1 - sc0 < patch + 4:
        return MatchSet.empty("reseed", cell_id=cell_id, reason="cell too small")

    sub = src_repr[sr0:sr1, sc0:sc1]
    pc = phase_congruency(sub)["pc"]
    kp = detect_pc_keypoints(pc, lattice=(2, 2), per_cell=max(2, n_candidates // 2),
                             border=patch // 4, min_pc=0.0)
    if len(kp) == 0:
        return MatchSet.empty("reseed", cell_id=cell_id, reason="no candidate in cell")

    kp_full = kp + np.array([sc0, sr0], dtype=np.float64)
    inside = (
        (kp_full[:, 0] >= c0) & (kp_full[:, 0] < c1) & (kp_full[:, 1] >= r0) & (kp_full[:, 1] < r1)
    )
    kp_full = kp_full[inside][:n_candidates]
    if len(kp_full) == 0:
        return MatchSet.empty("reseed", cell_id=cell_id, reason="no candidate inside the cell proper")

    predicted = apply_h(H_global, kp_full)
    refined, peaks, sharp = template_refine(
        src_repr, ref_repr, kp_full, predicted, window=2 * window_px, patch=patch, metric="cfog"
    )

    keep = peaks > min_peak
    if not keep.any():
        return MatchSet.empty("reseed", cell_id=cell_id, reason="no candidate passed the relaxed threshold")

    return MatchSet(
        kp_full[keep], refined[keep], np.clip(peaks[keep], 0.0, 1.0), track="reseed",
        meta={"cell_id": cell_id, "peak": peaks[keep], "sharpness": sharp[keep], "reseeded": True},
    )


def reseed_empty_cells(
    tiepoints: list[TiePoint],
    src_repr: np.ndarray,
    ref_repr: np.ndarray,
    lattice: Lattice,
    H_global: np.ndarray,
    max_passes: int = 2,
    window_px: int = 5,
    threshold_scale: float = 0.6,
    base_peak: float = 0.4,
) -> tuple[list[TiePoint], ReseedReport]:
    """Run the re-seeding loop until every valid cell is filled or the passes run out."""
    report = ReseedReport(coverage_before=lattice.coverage_ratio(tiepoints))
    next_tid = max((t.tid for t in tiepoints), default=-1) + 1
    added: list[TiePoint] = []

    for p in range(1, max_passes + 1):
        empty = lattice.empty_cells(tiepoints + added)
        if not empty:
            break
        report.passes = p
        report.cells_targeted += len(empty)
        # Each pass relaxes further: a cell that resisted the first attempt is one where
        # the texture is genuinely marginal.
        min_peak = base_peak * (threshold_scale**p)

        for cell_id in empty:
            ms = reseed_cell(src_repr, ref_repr, lattice, cell_id, H_global,
                             window_px=window_px, min_peak=min_peak)
            if ms.is_empty:
                continue
            best = int(np.argmax(ms.conf))
            added.append(TiePoint(
                tid=next_tid,
                src_sample=float(ms.kpts_src[best, 0]), src_line=float(ms.kpts_src[best, 1]),
                ref_sample=float(ms.kpts_ref[best, 0]), ref_line=float(ms.kpts_ref[best, 1]),
                conf=float(ms.conf[best]), track="reseed",
                sigma_x=np.nan, sigma_y=np.nan,
                inlier=True, reseeded=True, cell_id=cell_id,
            ))
            next_tid += 1
            report.cells_filled += 1
            report.points_added += 1

    out = tiepoints + added
    report.coverage_after = lattice.coverage_ratio(out)
    return out, report


def apply_quota(
    tiepoints: Sequence[TiePoint],
    lattice: Lattice,
    per_cell: int = 8,
    anms_radius_px: float = 24.0,
) -> list[TiePoint]:
    """Per-cell quota with ANMS, ranked by confidence / (1 + trace Sigma).

    The ranking is the specification's: a confident point with a large covariance loses
    to a slightly less confident one that is well localised, which is the right
    preference when the output is going to be used to fit a geometric model.
    """
    from setu.uniform.anms import anms_select

    lattice.assign(tiepoints)
    by_cell: dict[int, list[TiePoint]] = {}
    for t in tiepoints:
        by_cell.setdefault(t.cell_id, []).append(t)

    kept: list[TiePoint] = []
    for _cell, pts in by_cell.items():
        pts.sort(key=lambda t: t.quality, reverse=True)
        kept.extend(anms_select(pts, per_cell, min_radius_px=anms_radius_px))
    return kept
