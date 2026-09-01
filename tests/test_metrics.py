"""Metrics must recover a known synthetic shift, and must not flatter a fitted model."""

import numpy as np
import pytest

from setu.eval.metrics import (
    bootstrap_ci,
    ce90,
    cross_consistency,
    rmse,
    rmse_vs_truth,
    success_rate,
)


def test_rmse_of_identical_points_is_zero():
    pts = np.random.default_rng(0).uniform(0, 100, (50, 2))
    assert rmse(pts, pts) == pytest.approx(0.0, abs=1e-12)


def test_rmse_recovers_a_known_constant_shift():
    pts = np.random.default_rng(0).uniform(0, 100, (200, 2))
    for dx, dy in [(1.0, 0.0), (0.3, -0.4), (3.0, 4.0)]:
        assert rmse(pts + np.array([dx, dy]), pts) == pytest.approx(np.hypot(dx, dy), rel=1e-9)


def test_rmse_vs_truth_recovers_a_known_translation():
    """The headline accuracy metric must read a known transform error exactly."""
    H_true = np.eye(3)
    for shift in (0.05, 0.5, 2.0):
        H_est = np.array([[1, 0, shift], [0, 1, 0], [0, 0, 1]], dtype=float)
        assert rmse_vs_truth(H_est, H_true, (256, 256)) == pytest.approx(shift, rel=1e-6)


def test_rmse_vs_truth_is_infinite_for_a_degenerate_model():
    assert not np.isfinite(rmse_vs_truth(None, np.eye(3), (64, 64)))


def test_ce90_is_the_ninetieth_percentile():
    pts = np.zeros((100, 2))
    truth = np.zeros((100, 2))
    truth[:, 0] = np.arange(100) / 99.0
    assert ce90(pts, truth) == pytest.approx(np.percentile(np.arange(100) / 99.0, 90), rel=1e-9)


def test_success_rate_counts_thresholds():
    sr = success_rate([0.5, 2.0, 4.0, 20.0], [1.0, 3.0, 5.0, 10.0])
    assert sr["SR@1px"] == pytest.approx(0.25)
    assert sr["SR@3px"] == pytest.approx(0.50)
    assert sr["SR@5px"] == pytest.approx(0.75)


def test_bootstrap_interval_brackets_the_true_mean():
    rng = np.random.default_rng(1)
    covered = sum(
        1 for _ in range(60)
        if (ci := bootstrap_ci(rng.normal(5.0, 1.0, 60), n_boot=300, seed=int(rng.integers(1e6)))).lo
        <= 5.0 <= ci.hi
    )
    assert covered >= 51          # nominal 95%, allowing for finite-sample slack


def test_cross_consistency_is_zero_for_an_exact_inverse():
    H = np.array([[1.02, 0.01, 3.0], [-0.01, 0.99, -2.0], [1e-6, 2e-6, 1.0]])
    assert cross_consistency(H, np.linalg.inv(H)) == pytest.approx(0.0, abs=1e-9)


def test_cross_consistency_detects_a_biased_pair():
    H = np.eye(3)
    biased = np.array([[1, 0, 0.5], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert cross_consistency(H, biased) > 0.1
