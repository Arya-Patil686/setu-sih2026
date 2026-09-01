"""Sensor-realistic degradation for the controlled benchmark.

A rendered image is noise-free and band-limited only by the DEM. Matching two such
images measures the algorithm against a problem no payload actually poses. Each model
here reproduces the dominant artefact of one instrument, so that a benchmark result
transfers to the real archive instead of flattering it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class SensorProfile:
    """Radiometric and optical characteristics of one payload.

    `snr` is at the mid-scene signal level. `psf_sigma_px` is the combined optics and
    detector point-spread width. `tdi_smear_px` is the along-track smear from
    time-delay-integration line mismatch, which OHRC has and framing cameras do not.
    """

    name: str
    snr: float
    psf_sigma_px: float
    read_noise_frac: float
    tdi_smear_px: float = 0.0
    stripe_frac: float = 0.0
    quantisation_bits: int = 12


#: Per-payload profiles. OHRC's TDI smear and IIRS's column striping are the two
#: artefacts that most change matcher behaviour, so they are modelled explicitly
#: rather than folded into a single noise term.
PROFILES: dict[str, SensorProfile] = {
    "OHRC": SensorProfile("OHRC", snr=100.0, psf_sigma_px=0.55, read_noise_frac=0.004, tdi_smear_px=0.7, quantisation_bits=10),
    "TMC2_NADIR": SensorProfile("TMC2_NADIR", snr=90.0, psf_sigma_px=0.50, read_noise_frac=0.005, quantisation_bits=10),
    "TMC2_FORE": SensorProfile("TMC2_FORE", snr=80.0, psf_sigma_px=0.55, read_noise_frac=0.006, quantisation_bits=10),
    "TMC2_AFT": SensorProfile("TMC2_AFT", snr=80.0, psf_sigma_px=0.55, read_noise_frac=0.006, quantisation_bits=10),
    "IIRS": SensorProfile("IIRS", snr=35.0, psf_sigma_px=0.70, read_noise_frac=0.02, stripe_frac=0.035, quantisation_bits=12),
    "NAC_L": SensorProfile("NAC_L", snr=110.0, psf_sigma_px=0.45, read_noise_frac=0.003, tdi_smear_px=0.3, quantisation_bits=12),
    "NAC_R": SensorProfile("NAC_R", snr=110.0, psf_sigma_px=0.45, read_noise_frac=0.003, tdi_smear_px=0.3, quantisation_bits=12),
    "KAGUYA_TC": SensorProfile("KAGUYA_TC", snr=95.0, psf_sigma_px=0.50, read_noise_frac=0.005, quantisation_bits=10),
    "WAC": SensorProfile("WAC", snr=70.0, psf_sigma_px=0.60, read_noise_frac=0.008, quantisation_bits=10),
    "SYNTHETIC": SensorProfile("SYNTHETIC", snr=1e6, psf_sigma_px=0.0, read_noise_frac=0.0),
}


def photon_read_noise(
    image: np.ndarray,
    snr: float,
    read_noise_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shot noise scaling as sqrt(signal), plus a signal-independent read term.

    Shot noise is the reason low-sun lunar imagery is hard: the shadowed side of a
    crater carries almost no signal, so its noise is proportionally enormous and every
    gradient-based descriptor keys onto noise there.
    """
    img = np.asarray(image, dtype=np.float32)
    level = float(np.mean(np.abs(img))) or 1.0
    electrons = np.clip(img / level, 0.0, None) * (snr**2)
    shot = rng.normal(0.0, np.sqrt(np.maximum(electrons, 1.0))) / (snr**2) * level
    read = rng.normal(0.0, read_noise_frac * level, img.shape)
    return (img + shot + read).astype(np.float32)


def tdi_smear(image: np.ndarray, smear_px: float, axis: int = 0) -> np.ndarray:
    """Along-track smear from time-delay-integration line mismatch.

    A TDI sensor sums charge across many stages as the ground track moves beneath it. Any
    mismatch between the line rate and the true ground velocity smears the scene
    along-track by a fraction of a pixel per stage, which shows up as a directional blur.

    Modelled as a Gaussian of standard deviation `smear_px / sqrt(12)`, which is the
    standard deviation of a box of that width. This matters more than the choice of
    kernel shape suggests: the obvious implementation, a box filter of `round(smear_px)`
    samples, is *even*-width for typical sub-pixel smears, and an even box filter does not
    blur an image - it translates it half a pixel. A half-pixel translation applied only
    to the reference is indistinguishable from a registration error, and would have
    quietly become the noise floor of every sub-pixel measurement in this benchmark.
    """
    if smear_px <= 0:
        return np.asarray(image, dtype=np.float32)
    sigma = smear_px / np.sqrt(12.0)
    sigmas = [0.0, 0.0]
    sigmas[axis] = sigma
    return gaussian_filter(np.asarray(image, dtype=np.float32), sigma=sigmas).astype(np.float32)


def column_stripe(image: np.ndarray, stripe_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Fixed-pattern column gain and offset, the pushbroom spectrometer signature."""
    if stripe_frac <= 0:
        return np.asarray(image, dtype=np.float32)
    img = np.asarray(image, dtype=np.float32)
    level = float(np.mean(np.abs(img))) or 1.0
    gain = 1.0 + rng.normal(0.0, stripe_frac, (1, img.shape[1]))
    offset = rng.normal(0.0, stripe_frac * level, (1, img.shape[1]))
    return (img * gain + offset).astype(np.float32)


def quantise(image: np.ndarray, bits: int) -> np.ndarray:
    """Digitise to the payload's bit depth, preserving the physical range."""
    if bits <= 0 or bits >= 32:
        return np.asarray(image, dtype=np.float32)
    img = np.asarray(image, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return img
    levels = 2**bits - 1
    return (np.round((img - lo) / (hi - lo) * levels) / levels * (hi - lo) + lo).astype(np.float32)


def degrade(
    image: np.ndarray,
    sensor: str = "SYNTHETIC",
    rng: np.random.Generator | None = None,
    extra_psf_px: float = 0.0,
) -> np.ndarray:
    """Apply one payload's full degradation chain, in acquisition order.

    Optics first, then TDI smear, then the detector's noise and fixed pattern, then
    quantisation. Reordering these gives subtly wrong noise statistics - blurring after
    adding noise, for instance, correlates the noise and makes matching look easier.
    """
    rng = rng or np.random.default_rng(0)
    prof = PROFILES.get(sensor, PROFILES["SYNTHETIC"])
    img = np.asarray(image, dtype=np.float32)

    sigma = float(np.hypot(prof.psf_sigma_px, extra_psf_px))
    if sigma > 0:
        img = gaussian_filter(img, sigma).astype(np.float32)
    img = tdi_smear(img, prof.tdi_smear_px)
    if prof.snr < 1e5:
        img = photon_read_noise(img, prof.snr, prof.read_noise_frac, rng)
    img = column_stripe(img, prof.stripe_frac, rng)
    return quantise(img, prof.quantisation_bits)


def thermal_like(image: np.ndarray, dem: np.ndarray, gsd_m: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """A crude stand-in for an IIRS long-wave band, for the cross-modal path.

    Beyond about 2.5 um the scene is thermal emission rather than reflected sunlight:
    sunlit slopes are warm and therefore bright, but shadowed ground stays warm too
    because it was illuminated minutes earlier, so contrast collapses and a slow
    thermal-inertia component dominates. Inverting the shading term and adding that
    low-frequency field reproduces the qualitative failure mode without pretending to
    be a radiative transfer model.
    """
    from scipy.ndimage import gaussian_filter as gf

    rng = rng or np.random.default_rng(0)
    img = np.asarray(image, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    norm = (img - lo) / max(1e-9, hi - lo)

    inverted = 1.0 - norm
    inertia = gf(np.asarray(dem, dtype=np.float32), sigma=max(2.0, 120.0 / max(gsd_m, 1e-6)))
    inertia = (inertia - inertia.mean()) / (inertia.std() or 1.0)

    out = 0.45 * inverted + 0.35 * norm + 0.20 * (0.5 + 0.15 * inertia)
    return (out * (hi - lo) + lo).astype(np.float32)
