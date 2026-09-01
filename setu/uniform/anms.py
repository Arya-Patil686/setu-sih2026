"""S6 - adaptive non-maximal suppression within a lattice cell.

Applying the per-cell quota by simply taking the top-q points by quality would put all
q of them on the one crisp crater that dominates the cell. ANMS instead ranks points by
their suppression radius - the distance to the nearest point of meaningfully higher
quality - so the surviving set is spread across the cell by construction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from setu.types import TiePoint


def suppression_radii(points: np.ndarray, quality: np.ndarray, robust_ratio: float = 0.9) -> np.ndarray:
    """Distance from each point to the nearest point of significantly higher quality.

    The `robust_ratio` is the standard Brown et al. relaxation: a neighbour has to be
    better by a clear margin to suppress, otherwise two nearly equal detections suppress
    each other and both radii collapse.

    Points that nothing dominates are assigned the diameter of the point set rather than
    infinity. Infinity looks harmless and is not: with a relaxed ratio a fifth of the
    points end up undominated, they all tie at infinity, and the sort then orders them
    arbitrarily - which reproduces whatever spatial bias the input had, and can leave the
    selection *more* clustered than picking at random.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = np.asarray(quality, dtype=np.float64).ravel()
    n = len(p)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.array([1.0])

    diameter = float(np.hypot(np.ptp(p[:, 0]), np.ptp(p[:, 1]))) or 1.0
    order = np.argsort(q)[::-1]
    radii = np.full(n, diameter)

    # Walking from the strongest point outwards, every candidate only has to be tested
    # against the points already placed, since anything weaker can never dominate it.
    placed = np.empty((n, 2), dtype=np.float64)
    placed_q = np.empty(n, dtype=np.float64)
    count = 0

    for idx in order:
        if count:
            better = placed_q[:count] > q[idx] / max(robust_ratio, 1e-9)
            if better.any():
                d = np.hypot(*(placed[:count][better] - p[idx]).T)
                radii[idx] = float(min(d.min(), diameter))
        placed[count] = p[idx]
        placed_q[count] = q[idx]
        count += 1

    return radii


def anms_select(
    tiepoints: Sequence[TiePoint],
    n_keep: int,
    min_radius_px: float = 0.0,
    robust_ratio: float = 0.9,
    quality_weight: float = 0.5,
) -> list[TiePoint]:
    """Keep `n_keep` points, spread across the cell rather than piled on its best feature.

    Implemented as quality-seeded farthest-point selection: the strongest point is taken
    first, and each subsequent pick maximises its distance to everything already chosen,
    weighted mildly by quality.

    The textbook suppression-radius ranking is computed too and is what
    `suppression_radii` reports, but it is not what selects here, and the reason is worth
    recording. Brown's formulation assumes a detector response with a wide dynamic range,
    so that almost every point has some clearly stronger neighbour and its radius becomes
    a measure of local density. The quality used here is confidence / (1 + trace Sigma),
    which spans well under one order of magnitude, so a large fraction of points end up
    with no dominating neighbour at all, tie at the maximum radius, and get ordered by
    something that has nothing to do with where they are. On a clustered input that ranks
    *worse* than choosing at random. Farthest-point selection has no such failure mode:
    the criterion is spatial by construction.
    """
    pts = list(tiepoints)
    if len(pts) <= n_keep:
        return pts

    coords = np.array([[t.src_sample, t.src_line] for t in pts], dtype=np.float64)
    quality = np.array([t.quality for t in pts], dtype=np.float64)

    span = float(np.ptp(quality))
    weight = (quality - quality.min()) / span if span > 1e-12 else np.ones_like(quality)
    weight = weight**quality_weight

    selected = [int(np.argmax(quality))]
    dist = np.hypot(*(coords - coords[selected[0]]).T)

    while len(selected) < n_keep:
        score = dist * (0.5 + 0.5 * weight)
        score[selected] = -np.inf
        nxt = int(np.argmax(score))
        if not np.isfinite(score[nxt]) or dist[nxt] <= 0:
            break
        selected.append(nxt)
        dist = np.minimum(dist, np.hypot(*(coords - coords[nxt]).T))

    chosen = [pts[i] for i in selected]
    if min_radius_px > 0 and len(chosen) > 1:
        chosen = enforce_min_distance(chosen, min_radius_px)
    return chosen


def enforce_min_distance(tiepoints: Sequence[TiePoint], min_radius_px: float) -> list[TiePoint]:
    """Greedily drop points closer than `min_radius_px` to an already-kept better point."""
    pts = sorted(tiepoints, key=lambda t: t.quality, reverse=True)
    if len(pts) < 2:
        return list(pts)
    coords = np.array([[t.src_sample, t.src_line] for t in pts])
    tree = cKDTree(coords)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(coords[i], min_radius_px):
            if j > i:
                keep[j] = False
    return [t for t, k in zip(pts, keep) if k]
