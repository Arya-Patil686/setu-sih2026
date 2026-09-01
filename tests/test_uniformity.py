"""Uniformity statistics must be calibrated against point processes with known behaviour."""

import numpy as np
import pytest

from setu.eval.metrics import clark_evans, coverage_ratio, occupancy_chi_square
from setu.types import TiePoint
from setu.uniform.anms import anms_select, suppression_radii
from setu.uniform.lattice import auto_lattice_size, build_lattice
from setu.uniform.stats import uniformity_report

AREA = 512 * 512


def _poisson(n=400, seed=0):
    return np.random.default_rng(seed).uniform(0, 512, (n, 2))


def _grid(n_side=20):
    xs = np.linspace(12, 500, n_side)
    return np.stack(np.meshgrid(xs, xs), -1).reshape(-1, 2)


def _cluster(n=400, seed=0):
    return np.random.default_rng(seed).normal(256, 18, (n, 2))


def test_clark_evans_is_one_for_a_poisson_process():
    """R = 1 is the definition of complete spatial randomness; this calibrates the metric."""
    rs = [clark_evans(_poisson(400, s), AREA)["clark_evans_R"] for s in range(40)]
    assert np.mean(rs) == pytest.approx(1.0, abs=0.05)


def test_clark_evans_exceeds_one_for_a_regular_grid():
    r = clark_evans(_grid(), AREA)["clark_evans_R"]
    assert r > 1.5
    assert r <= 2.16          # 2.149 is the theoretical maximum for a hexagonal lattice


def test_clark_evans_is_below_one_for_a_cluster():
    assert clark_evans(_cluster(), AREA)["clark_evans_R"] < 0.5


def test_chi_square_does_not_reject_a_uniform_process():
    ps = [occupancy_chi_square(_poisson(600, s), (512, 512), (8, 8))["chi2_p"] for s in range(30)]
    assert np.mean([p > 0.05 for p in ps]) > 0.8


def test_chi_square_rejects_a_clustered_process():
    assert occupancy_chi_square(_cluster(), (512, 512), (8, 8))["chi2_p"] < 1e-6


def test_coverage_is_total_for_a_dense_uniform_process():
    assert coverage_ratio(_poisson(2000), (512, 512), (8, 8)) == pytest.approx(1.0)


def test_coverage_is_low_for_a_cluster():
    assert coverage_ratio(_cluster(), (512, 512), (8, 8)) < 0.25


def test_auto_lattice_scales_with_the_target_count():
    assert auto_lattice_size(400) == (10, 10)
    assert auto_lattice_size(100) == (5, 5)
    assert auto_lattice_size(1_000_000) == (16, 16)      # clamped


def test_lattice_marks_cells_outside_the_overlap_invalid():
    overlap = np.zeros((512, 512), bool)
    overlap[0:256, 0:256] = True
    lattice = build_lattice((512, 512), (8, 8), overlap)
    assert lattice.n_valid == 16                          # exactly one quadrant
    assert lattice.overlap_area_px == pytest.approx(256 * 256)


def test_suppression_radius_is_maximal_for_the_strongest_point():
    """Undominated points take the set diameter, never infinity - see `suppression_radii`."""
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    radii = suppression_radii(pts, np.array([1.0, 0.5, 0.4]))
    assert np.isfinite(radii).all()
    assert radii[0] == radii.max()
    assert radii[1] < radii[0]


def _tiepoints(coords, seed=0):
    rng = np.random.default_rng(seed)
    return [
        TiePoint(i, float(y), float(x), float(y), float(x),
                 conf=float(rng.uniform(0.5, 1.0)), sigma_x=0.1, sigma_y=0.1)
        for i, (x, y) in enumerate(coords)
    ]


def test_anms_beats_a_size_matched_random_subsample():
    """The fair comparison, because Clark-Evans R depends on the point count.

    Comparing R before and after selection would be comparing 200 points with 40, and R
    falls with N for the same spatial pattern. The question ANMS actually has to answer is
    whether *its* 40 points are better spread than 40 taken at random from the same set.
    """
    rng = np.random.default_rng(0)
    coords = np.vstack([_cluster(180, 1), rng.uniform(20, 490, (20, 2))])
    selected = anms_select(_tiepoints(coords), 40, min_radius_px=0)
    picked = np.array([[t.src_sample, t.src_line] for t in selected])

    r_anms = clark_evans(picked, AREA)["clark_evans_R"]
    r_random = np.mean([
        clark_evans(coords[np.random.default_rng(s).choice(len(coords), len(picked), replace=False)],
                    AREA)["clark_evans_R"]
        for s in range(40)
    ])
    assert r_anms > r_random


def test_quota_with_a_lattice_delivers_coverage():
    """How S6 actually uses ANMS: a per-cell quota over the lattice, not a global top-n."""
    from setu.uniform.reseed import apply_quota

    rng = np.random.default_rng(0)
    coords = np.vstack([_cluster(300, 1), rng.uniform(10, 500, (120, 2))])
    pts = _tiepoints(coords)
    lattice = build_lattice((512, 512), (8, 8))

    kept = apply_quota(pts, lattice, per_cell=4, anms_radius_px=8.0)
    picked = np.array([[t.src_sample, t.src_line] for t in kept])

    assert len(kept) < len(pts)
    # The quota caps how much the cluster can dominate, so the survivors spread out.
    assert clark_evans(picked, AREA)["clark_evans_R"] > clark_evans(coords, AREA)["clark_evans_R"]
    assert coverage_ratio(picked, (512, 512), (8, 8)) >= coverage_ratio(coords, (512, 512), (8, 8))


def test_uniformity_report_flags_its_own_targets():
    lattice = build_lattice((512, 512), (8, 8))
    pts = [TiePoint(i, float(y), float(x), float(y), float(x), conf=0.9, sigma_x=0.1, sigma_y=0.1)
           for i, (x, y) in enumerate(_poisson(600))]
    rep = uniformity_report(pts, lattice)
    assert rep["coverage_pass"] is True
    assert set(rep) >= {"coverage_ratio", "chi2_p", "clark_evans_R",
                        "coverage_pass", "chi2_pass", "clark_evans_pass"}
