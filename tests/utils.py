"""Shared utility functions for ripple tests.

This module contains common utilities used across different test categories:
- Frequency grid construction
- Waveform detection and loading
- LALSuite waveform generation (when available)
- Match computation and inner products
- PSD loading
- Random parameter generation
"""

import os
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ripplegw import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.constants import PI

# Check if LALSuite is available
try:
    import lal
    import lalsimulation as lalsim

    LAL_AVAILABLE = True
except ImportError:
    LAL_AVAILABLE = False


def check_lal_available():
    """Check if LALSuite is available and raise ImportError if not.

    Raises:
        ImportError: If LALSuite is not installed.
    """
    if not LAL_AVAILABLE:
        raise ImportError(
            "LALSuite is required for cross-validation tests. "
            "Install it with: uv sync --all-extras --dev"
        )


def get_freqs(f_l: float, f_u: float, f_sampling: float, T: float) -> jnp.ndarray:
    """Construct a frequency grid for waveform generation.

    Args:
        f_l: Lower frequency bound (Hz).
        f_u: Upper frequency bound (Hz).
        f_sampling: Sampling frequency (Hz).
        T: Duration (seconds).

    Returns:
        Frequency array in the range (f_l, f_u).
    """
    delta_t = 1 / f_sampling
    tlen = int(round(T / delta_t))
    freqs = np.fft.rfftfreq(tlen, delta_t)
    fs = freqs[(freqs > f_l) & (freqs < f_u)]
    return jnp.array(fs)


def check_is_tidal(waveform_name: str) -> bool:
    """Check if the given waveform includes tidal effects.

    Args:
        waveform_name: Name of the waveform approximant.

    Returns:
        True if the waveform includes tidal effects, False otherwise.

    Raises:
        ValueError: If the waveform is not supported.
    """
    bns_waveforms = ["IMRPhenomD_NRTidalv2", "TaylorF2", "IMRPhenomXAS_NRTidalv3"]
    bbh_waveforms = ["IMRPhenomD", "IMRPhenomXAS", "IMRPhenomPv2", "SineGaussian"]

    all_waveforms = bns_waveforms + bbh_waveforms
    if waveform_name not in all_waveforms:
        raise ValueError(f"Waveform approximant {waveform_name} not supported by ripple")

    return waveform_name in bns_waveforms


def check_is_precessing(waveform_name: str) -> bool:
    """Check if the given waveform includes precession.

    Args:
        waveform_name: Name of the waveform approximant.

    Returns:
        True if the waveform includes precession, False otherwise.
    """
    precessing_waveforms = ["IMRPhenomPv2"]
    return waveform_name in precessing_waveforms


def get_jitted_waveform(waveform_name: str, fs: jnp.ndarray, f_ref: float):
    """Get a JIT-compiled waveform function.

    Args:
        waveform_name: Name of the waveform approximant.
        fs: Frequency array.
        f_ref: Reference frequency (Hz).

    Returns:
        JIT-compiled waveform function that takes theta and returns hp.

    Raises:
        ValueError: If the waveform is not supported.
    """
    if waveform_name == "IMRPhenomD":
        from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc as waveform_generator

        @jax.jit
        def waveform(theta):
            hp, _ = waveform_generator(fs, theta, f_ref)
            return hp

    elif waveform_name == "IMRPhenomD_NRTidalv2":
        from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import (
            gen_IMRPhenomD_NRTidalv2_hphc as waveform_generator,
        )

        @jax.jit
        def waveform(theta):
            hp, _ = waveform_generator(fs, theta, f_ref)
            return hp

    elif waveform_name == "IMRPhenomXAS":
        from ripplegw.waveforms.IMRPhenomXAS import (
            gen_IMRPhenomXAS_hphc as waveform_generator,
        )

        @jax.jit
        def waveform(theta):
            hp, _ = waveform_generator(fs, theta, f_ref)
            return hp

    elif waveform_name == "IMRPhenomXAS_NRTidalv3":
        from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import (
            gen_IMRPhenomXAS_NRTidalv3_hphc as waveform_generator,
        )

        @jax.jit
        def waveform(theta):
            hp, _ = waveform_generator(fs, theta, f_ref)
            return hp

    elif waveform_name == "TaylorF2":
        from ripplegw.waveforms.TaylorF2 import gen_TaylorF2_hphc as waveform_generator

        @jax.jit
        def waveform(theta):
            hp, _ = waveform_generator(fs, theta, f_ref)
            return hp

    elif waveform_name == "IMRPhenomPv2":
        from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2

        @jax.jit
        def waveform(theta):
            hp, hc = gen_IMRPhenomPv2(fs, theta, f_ref)
            return hp

    elif waveform_name == "SineGaussian":
        from ripplegw.waveforms.SineGaussian import gen_SineGaussian_hphc

        @jax.jit
        def waveform(theta):
            hp, _ = gen_SineGaussian_hphc(fs, theta)
            return hp

    else:
        raise ValueError(f"Waveform approximant {waveform_name} not supported by ripple")

    return waveform


def get_lal_waveform(
    theta: np.ndarray,
    waveform_name: str,
    f_l: float,
    f_u: float,
    df: float,
    f_ref: float,
    is_tidal: bool,
    is_precessing: bool,
) -> np.ndarray:
    """Generate a waveform using LALSuite.

    Args:
        theta: Parameter array. For non-precessing: [m1, m2, s1z, s2z, (l1, l2), dist, tc, phic, inc].
                For precessing: [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist, tc, phic, inc].
        waveform_name: Name of the waveform approximant.
        f_l: Lower frequency bound (Hz).
        f_u: Upper frequency bound (Hz).
        df: Frequency spacing (Hz).
        f_ref: Reference frequency (Hz).
        is_tidal: Whether the waveform includes tidal effects.
        is_precessing: Whether the waveform includes precession.

    Returns:
        LAL waveform strain (hp) evaluated on the frequency grid.

    Raises:
        ImportError: If LALSuite is not available.
    """
    check_lal_available()
    
    # Convert JAX arrays to Python floats if necessary
    f_l = float(f_l)
    f_u = float(f_u)
    df = float(df)
    f_ref = float(f_ref)

    approximant = lalsim.SimInspiralGetApproximantFromString(waveform_name)

    if is_precessing:
        # Precessing waveform: theta = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist, tc, phic, inc]
        m1_kg = theta[0] * lal.MSUN_SI
        m2_kg = theta[1] * lal.MSUN_SI
        s1x, s1y, s1z = theta[2], theta[3], theta[4]
        s2x, s2y, s2z = theta[5], theta[6], theta[7]
        distance = theta[8] * 1e6 * lal.PC_SI
        phi_ref = theta[10]
        inclination = theta[11]

        hp, _ = lalsim.SimInspiralChooseFDWaveform(
            m1_kg,
            m2_kg,
            s1x,
            s1y,
            s1z,
            s2x,
            s2y,
            s2z,
            distance,
            inclination,
            phi_ref,
            0,
            0,
            0,
            df,
            f_l,
            f_u,
            f_ref,
            None,
            approximant,
        )
    else:
        # Non-precessing waveform: theta = [m1, m2, s1z, s2z, (l1, l2), dist, tc, phic, inc]
        if is_tidal:
            m1_kg = theta[0] * lal.MSUN_SI
            m2_kg = theta[1] * lal.MSUN_SI
            s1z = theta[2]
            s2z = theta[3]
            l1 = theta[4]
            l2 = theta[5]
            distance = theta[6] * 1e6 * lal.PC_SI
            phi_ref = theta[8]
            inclination = theta[9]

            laldict = lal.CreateDict()
            lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, l1)
            lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, l2)
            quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1)
            quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2)
            lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
            lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)
        else:
            m1_kg = theta[0] * lal.MSUN_SI
            m2_kg = theta[1] * lal.MSUN_SI
            s1z = theta[2]
            s2z = theta[3]
            distance = theta[4] * 1e6 * lal.PC_SI
            phi_ref = theta[6]
            inclination = theta[7]
            laldict = None

        hp, _ = lalsim.SimInspiralChooseFDWaveform(
            m1_kg,
            m2_kg,
            0.0,
            0.0,
            s1z,
            0.0,
            0.0,
            s2z,
            distance,
            inclination,
            phi_ref,
            0,
            0,
            0,
            df,
            f_l,
            f_u,
            f_ref,
            laldict,
            approximant,
        )

    # Extract the waveform data and mask to the frequency range
    freqs_lal = np.arange(len(hp.data.data)) * df
    mask_lal = (freqs_lal > f_l) & (freqs_lal < f_u)
    hp_lalsuite = hp.data.data[mask_lal]

    return hp_lalsuite


def get_nyquist_mask(frequencies: jnp.ndarray, n_bins: int = 2) -> jnp.ndarray:
    """Create a mask that zeros the last n_bins frequency bins near Nyquist.

    LAL's behavior at the Nyquist frequency boundary is inconsistent - it sometimes
    zeros 1 bin, sometimes 2 bins depending on the waveform parameters. To ensure
    a fair comparison, we apply the same mask to both LAL and Ripple waveforms.

    Args:
        frequencies: Frequency array.
        n_bins: Number of bins to zero (default: 2).

    Returns:
        Mask array with 1.0 everywhere except the last n_bins (which are 0.0).
    """
    n_freqs = len(frequencies)
    return jnp.where(jnp.arange(n_freqs) < n_freqs - n_bins, 1.0, 0.0)


def noise_weighted_inner_product(
    h1: jnp.ndarray, h2: jnp.ndarray, psd: jnp.ndarray, frequencies: jnp.ndarray
) -> float:
    """Compute the noise-weighted inner product between two waveforms.

    Args:
        h1: First waveform.
        h2: Second waveform.
        psd: Power spectral density.
        frequencies: Frequency array.

    Returns:
        Noise-weighted inner product (h1|h2).
    """
    from jax.scipy.integrate import trapezoid

    integrand = jnp.conj(h1) * h2 / psd
    return 4 * trapezoid(integrand, x=frequencies, axis=-1).real


def compute_match(
    h1: jnp.ndarray, h2: jnp.ndarray, psd: jnp.ndarray, frequencies: jnp.ndarray
) -> float:
    """Compute the match between two waveforms.

    The match is defined as the normalized inner product:
        match = (h1|h2) / sqrt((h1|h1) * (h2|h2))

    Args:
        h1: First waveform.
        h2: Second waveform.
        psd: Power spectral density.
        frequencies: Frequency array.

    Returns:
        Match value (scalar between 0 and 1).
    """
    h1_sq = noise_weighted_inner_product(h1, h1, psd, frequencies)
    h2_sq = noise_weighted_inner_product(h2, h2, psd, frequencies)
    h1_h2 = noise_weighted_inner_product(h1, h2, psd, frequencies)
    match = h1_h2 / jnp.sqrt(h1_sq * h2_sq)
    return match.real


def load_psd(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a PSD file from the tests/psds/ directory.

    Args:
        name: Name of the PSD file (e.g., 'psd.txt', 'ET-D-psd.txt').

    Returns:
        Tuple of (frequencies, PSD values).

    Raises:
        FileNotFoundError: If the PSD file does not exist.
    """
    # Try loading from current directory first, then relative to tests directory
    psd_path = Path(name)
    if not psd_path.exists():
        psd_path = Path(__file__).parent / "psds" / name
    if not psd_path.exists():
        raise FileNotFoundError(f"PSD file not found: {name}")

    freqs, *psd_arrs = np.loadtxt(psd_path, unpack=True)
    psd = psd_arrs[0]
    return freqs, psd


def generate_random_params(
    n: int,
    bounds: dict,
    is_tidal: bool = False,
    is_precessing: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate random waveform parameters.

    Args:
        n: Number of parameter sets to generate.
        bounds: Dictionary of parameter bounds with keys:
            - "m": [m_min, m_max] for masses (solar masses)
            - "chi": [chi_min, chi_max] for spins
            - "lambda": [lambda_min, lambda_max] for tidal parameters (if is_tidal)
            - "d_L": [d_min, d_max] for distance (Mpc)
        is_tidal: Whether to include tidal parameters.
        is_precessing: Whether to include precessing spin parameters.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n, n_params) with random parameter values.
        For non-precessing, non-tidal: [m1, m2, s1z, s2z, dist, tc, phic, inc]
        For non-precessing, tidal: [m1, m2, s1z, s2z, l1, l2, dist, tc, phic, inc]
        For precessing: [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist, tc, phic, inc]
    """
    if seed is not None:
        np.random.seed(seed)

    m1 = np.random.uniform(bounds["m"][0], bounds["m"][1], n)
    m2 = np.random.uniform(bounds["m"][0], bounds["m"][1], n)

    if is_precessing:
        # Precessing: generate full spin vectors
        # Sample spin magnitudes uniformly
        chi1_mag = np.random.uniform(bounds["chi"][0], bounds["chi"][1], n)
        chi2_mag = np.random.uniform(bounds["chi"][0], bounds["chi"][1], n)
        # Sample angles uniformly on sphere
        theta1 = np.arccos(np.random.uniform(-1, 1, n))
        theta2 = np.arccos(np.random.uniform(-1, 1, n))
        phi1 = np.random.uniform(0, 2 * PI, n)
        phi2 = np.random.uniform(0, 2 * PI, n)
        # Convert to Cartesian
        s1x = chi1_mag * np.sin(theta1) * np.cos(phi1)
        s1y = chi1_mag * np.sin(theta1) * np.sin(phi1)
        s1z = chi1_mag * np.cos(theta1)
        s2x = chi2_mag * np.sin(theta2) * np.cos(phi2)
        s2y = chi2_mag * np.sin(theta2) * np.sin(phi2)
        s2z = chi2_mag * np.cos(theta2)
    else:
        # Aligned spin: only z-components
        s1z = np.random.uniform(bounds["chi"][0], bounds["chi"][1], n)
        s2z = np.random.uniform(bounds["chi"][0], bounds["chi"][1], n)

    if is_tidal:
        l1 = np.random.uniform(bounds["lambda"][0], bounds["lambda"][1], n)
        l2 = np.random.uniform(bounds["lambda"][0], bounds["lambda"][1], n)

    dist_mpc = np.random.uniform(bounds["d_L"][0], bounds["d_L"][1], n)
    tc = np.zeros_like(dist_mpc)
    inclination = np.random.uniform(0, PI, n)
    phi_ref = np.random.uniform(0, 2 * PI, n)

    # Build parameter array
    if is_precessing:
        theta = np.array(
            [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phi_ref, inclination]
        ).T
    elif is_tidal:
        theta = np.array(
            [m1, m2, s1z, s2z, l1, l2, dist_mpc, tc, phi_ref, inclination]
        ).T
    else:
        theta = np.array([m1, m2, s1z, s2z, dist_mpc, tc, phi_ref, inclination]).T

    # Ensure m1 >= m2
    if not is_precessing:
        # Non-precessing: swap masses and spins
        booleans = theta[:, 0] < theta[:, 1]
        booleans = np.repeat(booleans[:, np.newaxis], theta.shape[1], axis=1)
        if is_tidal:
            theta = np.where(
                booleans, theta[:, [1, 0, 3, 2, 5, 4, 6, 7, 8, 9]], theta
            )
        else:
            theta = np.where(booleans, theta[:, [1, 0, 3, 2, 4, 5, 6, 7]], theta)
    else:
        # Precessing: swap masses and all spin components
        booleans = theta[:, 0] < theta[:, 1]
        booleans = np.repeat(booleans[:, np.newaxis], theta.shape[1], axis=1)
        theta = np.where(
            booleans, theta[:, [1, 0, 5, 6, 7, 2, 3, 4, 8, 9, 10, 11]], theta
        )

    return theta
