"""S6 - the three uniformity statistics, computed over the true overlap.

Almost every competing team will report a match count. The problem statement asks for a
uniform distribution, which a count cannot express: 400 points clustered on one bright
crater and 400 points spread across the scene are the same number and completely
different products.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from setu.eval.metrics import clark_evans
from setu.types import TiePoint
from setu.uniform.lattice import Lattice


def uniformity_report(
    tiepoints: Sequence[TiePoint],
    lattice: Lattice,
    inliers_only: bool = True,
) -> dict[str, Any]:
    """Coverage ratio, occupancy chi-square and the Clark-Evans index, with their targets.

    Each statistic is reported next to the target the specification sets for it, so the
    output states whether it passed rather than leaving that to the reader.
    """
    pts = [t for t in tiepoints if (t.inlier or not inliers_only)]
    coords = np.array([[t.src_sample, t.src_line] for t in pts]) if pts else np.zeros((0, 2))

    coverage = lattice.coverage_ratio(pts)
    occ = lattice.occupancy(pts)
    valid_occ = occ[lattice.valid_cells]

    chi = _chi_square_valid(valid_occ)
    ce = clark_evans(coords, lattice.overlap_area_px)

    return {
        "n_points": len(pts),
        "lattice": [lattice.m, lattice.n],
        "n_cells": lattice.n_cells,
        "n_valid_cells": lattice.n_valid,
        "coverage_ratio": round(float(coverage), 4),
        "coverage_target": 0.90,
        "coverage_pass": bool(coverage >= 0.90),
        "chi2": round(float(chi["chi2"]), 3),
        "chi2_p": round(float(chi["chi2_p"]), 5),
        "chi2_dof": chi["dof"],
        "chi2_pass": bool(np.isfinite(chi["chi2_p"]) and chi["chi2_p"] > 0.05),
        "clark_evans_R": round(float(ce["clark_evans_R"]), 4),
        "clark_evans_z": round(float(ce["clark_evans_z"]), 3),
        "clark_evans_target": [1.0, 1.4],
        "clark_evans_pass": bool(1.0 <= ce["clark_evans_R"] <= 1.4),
        "occupancy": occ.tolist(),
        "empty_valid_cells": len(lattice.empty_cells(pts)),
        "points_per_valid_cell": round(float(len(pts) / max(lattice.n_valid, 1)), 2),
    }


def _chi_square_valid(occupancy: np.ndarray) -> dict[str, float]:
    """Occupancy chi-square restricted to valid cells.

    Including cells that lie outside the overlap would guarantee a rejected null for
    every scene whose overlap is not rectangular, which is most of them.
    """
    occ = np.asarray(occupancy, dtype=float)
    n_cells = occ.size
    total = float(occ.sum())
    if n_cells < 2 or total == 0:
        return {"chi2": float("nan"), "chi2_p": float("nan"), "dof": max(n_cells - 1, 0)}
    from scipy import stats as sps

    expected = total / n_cells
    chi2 = float(((occ - expected) ** 2 / expected).sum())
    dof = n_cells - 1
    return {"chi2": chi2, "chi2_p": float(sps.chi2.sf(chi2, dof)), "dof": dof}
