"""Photometry sanity: limiting cases, monotonicity and energy behaviour."""

import numpy as np
import pytest

from setu.illum.reflectance import lambert, lommel_seeliger, lunar_lambert, mcewen_L
from setu.illum.shadow import horizon_shadow, surface_normals


def test_mcewen_L_is_one_at_zero_phase():
    """L(0) = 1 makes the Lunar-Lambert law reduce to pure Lambert at opposition."""
    assert mcewen_L(0.0) == pytest.approx(1.0)


def test_mcewen_L_decreases_with_phase_and_stays_bounded():
    g = np.linspace(0, 180, 181)
    L = mcewen_L(g)
    assert np.all((L >= 0.0) & (L <= 1.0))
    assert np.all(np.diff(L[:150]) <= 1e-9)


def test_lunar_lambert_limits_are_the_two_scattering_laws():
    """L is the mixing coefficient, and each end of its range is a named law.

    I = A * [ 2 L mu0/(mu0+mu) + (1-L) mu0 ]. At L = 1 the Lommel-Seeliger term is all
    that survives; at L = 0 it is pure Lambert. McEwen's L(0) = 1, so at zero phase the
    surface is fully Lommel-Seeliger - which is the convention ISIS uses, and getting it
    backwards would invert the limb darkening of every rendered reference.
    """
    albedo = 0.12
    for mu0, mu in ((0.2, 1.0), (0.5, 0.8), (1.0, 1.0)):
        # L(0) = 1 -> twice the Lommel-Seeliger term.
        assert lunar_lambert(mu0, mu, 0.0, albedo) == pytest.approx(
            2.0 * lommel_seeliger(mu0, mu, albedo), rel=1e-5)

    # Force L to zero by evaluating past the cubic's root: the law becomes Lambert.
    assert mcewen_L(160.0) == pytest.approx(0.0)
    for mu0 in (0.2, 0.5, 1.0):
        assert lunar_lambert(mu0, 1.0, 160.0, albedo) == pytest.approx(lambert(mu0, albedo), rel=1e-5)


def test_reflectance_vanishes_at_the_terminator():
    """Incidence of 90 degrees means no illumination, whatever the model."""
    assert lunar_lambert(0.0, 1.0, 60.0, 0.12) == pytest.approx(0.0, abs=1e-7)
    assert lommel_seeliger(0.0, 1.0, 0.12) == pytest.approx(0.0, abs=1e-7)


def test_reflectance_is_monotonic_in_incidence():
    mu0 = np.linspace(0.01, 1.0, 40)
    I = lunar_lambert(mu0, np.ones_like(mu0), 45.0, 0.12)
    assert np.all(np.diff(I) > 0)


def test_reflectance_scales_linearly_with_albedo():
    a = lunar_lambert(0.6, 0.9, 30.0, 0.10)
    b = lunar_lambert(0.6, 0.9, 30.0, 0.20)
    assert b == pytest.approx(2.0 * a, rel=1e-5)


def test_flat_terrain_casts_no_shadow():
    assert horizon_shadow(np.zeros((64, 64), np.float32), 10.0, 135.0, 20.0).sum() == 0


def test_shadow_falls_away_from_the_sun():
    """A cone lit from the east shadows its western side and not its eastern one."""
    yy, xx = np.mgrid[0:64, 0:64]
    cone = np.clip(300 - 12 * np.hypot(yy - 32, xx - 32), 0, None).astype(np.float32)
    east = horizon_shadow(cone, 10.0, 90.0, 20.0)
    assert east[:, :32].mean() > 0.2
    assert east[:, 32:].mean() == pytest.approx(0.0, abs=1e-6)
    # Reversing the sun reverses the shadow.
    west = horizon_shadow(cone, 10.0, 270.0, 20.0)
    assert west[:, 32:].mean() > 0.2


def test_shadow_shrinks_as_the_sun_rises():
    from setu.bench.terrain import synthetic_terrain

    dem = synthetic_terrain(256, 5.0, "highland", seed=1).dem
    fractions = [horizon_shadow(dem, 5.0, 135.0, e).mean() for e in (10, 20, 40, 70)]
    assert all(a >= b for a, b in zip(fractions, fractions[1:]))


def test_surface_normals_are_unit_length():
    from setu.bench.terrain import synthetic_terrain

    dem = synthetic_terrain(128, 5.0, "highland", seed=2).dem
    n = surface_normals(dem, 5.0)
    assert np.abs(np.linalg.norm(n, axis=-1) - 1.0).max() < 1e-5


def test_normals_of_flat_ground_point_straight_up():
    n = surface_normals(np.zeros((32, 32), np.float32), 10.0)
    assert n[..., 2].min() == pytest.approx(1.0, abs=1e-6)
