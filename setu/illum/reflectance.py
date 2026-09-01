"""Photometric models for lunar surface rendering.

Implements the Appendix formulae verbatim. The Lunar-Lambert law with McEwen's
empirical limb-darkening coefficient is the standard model for lunar photometric
correction and is what the re-illumination of S2a renders through.
"""

from __future__ import annotations

import numpy as np

#: McEwen (1991) limb-darkening coefficients, the standard cubic in phase angle in
#: degrees. These are the same values ISIS uses for its `lunarlambert` photometric
#: model, which keeps rendered references comparable with ISIS-processed products.
MCEWEN_COEFFS = (1.0, -1.9e-2, 2.42e-4, -1.46e-6)


def mcewen_L(phase_deg: float | np.ndarray) -> np.ndarray:
    """L(g), the Lunar-Lambert mixing coefficient, clamped to [0, 1].

    L = 1 at zero phase (pure Lambert) and falls towards 0 at high phase, where the
    Lommel-Seeliger term dominates. The clamp matters: the cubic goes negative past
    about 150 degrees, which would invert the shading.
    """
    g = np.asarray(phase_deg, dtype=np.float64)
    c0, c1, c2, c3 = MCEWEN_COEFFS
    L = c0 + c1 * g + c2 * g**2 + c3 * g**3
    return np.clip(L, 0.0, 1.0)


def lunar_lambert(
    mu0: np.ndarray,
    mu: np.ndarray,
    phase_deg: float | np.ndarray,
    albedo: float | np.ndarray = 0.12,
) -> np.ndarray:
    """I/F = A * [ 2*L(g)*mu0/(mu0 + mu) + (1 - L(g))*mu0 ].

    The first term is Lommel-Seeliger (single scattering off a rough, dark surface),
    the second is Lambert. `mu0` is cos(incidence) and `mu` is cos(emission), both
    already clamped at the terminator by the caller.
    """
    mu0 = np.clip(np.asarray(mu0, dtype=np.float32), 0.0, 1.0)
    mu = np.clip(np.asarray(mu, dtype=np.float32), 1e-6, 1.0)
    L = mcewen_L(phase_deg).astype(np.float32)
    ls = 2.0 * L * mu0 / (mu0 + mu)
    lam = (1.0 - L) * mu0
    return np.asarray(albedo, dtype=np.float32) * (ls + lam)


def lommel_seeliger(mu0: np.ndarray, mu: np.ndarray, albedo: float | np.ndarray = 0.12) -> np.ndarray:
    """Pure Lommel-Seeliger, the limiting case at high phase."""
    mu0 = np.clip(np.asarray(mu0, dtype=np.float32), 0.0, 1.0)
    mu = np.clip(np.asarray(mu, dtype=np.float32), 1e-6, 1.0)
    return np.asarray(albedo, dtype=np.float32) * mu0 / (mu0 + mu)


def lambert(mu0: np.ndarray, albedo: float | np.ndarray = 0.12) -> np.ndarray:
    """Pure Lambert, used only as a reference in the photometric unit tests."""
    return np.asarray(albedo, dtype=np.float32) * np.clip(np.asarray(mu0, dtype=np.float32), 0.0, 1.0)


def incidence_from_normals(normals: np.ndarray, sun_vec: np.ndarray) -> np.ndarray:
    """cos(incidence) per pixel from surface normals and one collimated sun vector.

    The Sun subtends about half a degree at the Moon and there is no atmosphere to
    scatter it, so a single direction for the whole patch is not an approximation
    worth improving on.
    """
    s = np.asarray(sun_vec, dtype=np.float32)
    s = s / max(1e-12, float(np.linalg.norm(s)))
    return np.einsum("ijk,k->ij", normals.astype(np.float32), s)


def emission_from_normals(normals: np.ndarray, view_vec: np.ndarray) -> np.ndarray:
    """cos(emission) per pixel for a given viewing direction."""
    v = np.asarray(view_vec, dtype=np.float32)
    v = v / max(1e-12, float(np.linalg.norm(v)))
    return np.einsum("ijk,k->ij", normals.astype(np.float32), v)


def view_vector(emission_deg: float, azimuth_deg: float = 0.0) -> np.ndarray:
    """Unit vector towards the sensor in the local East-North-Up frame."""
    e = np.radians(emission_deg)
    a = np.radians(azimuth_deg)
    return np.array([np.sin(e) * np.sin(a), np.sin(e) * np.cos(a), np.cos(e)], dtype=np.float64)
