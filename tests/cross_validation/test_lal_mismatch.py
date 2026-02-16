"""Cross-validation tests comparing ripple waveforms against LALSuite.

These tests compute noise-weighted mismatches between ripple and LALSuite
waveforms over randomly sampled parameter sets. They verify that ripple
achieves machine-precision agreement with LAL.

Nyquist Boundary Handling:
    LAL's behavior at the Nyquist frequency boundary is inconsistent - it
    sometimes zeros 1 bin, sometimes 2 bins depending on the waveform parameters.
    To ensure a fair comparison, we apply the same Nyquist mask (zeroing the
    last 2 frequency bins) to both LAL and ripple waveforms before computing
    the match.

Expected Results:
    - Non-tidal BBH (IMRPhenomD, IMRPhenomXAS): mismatch < 1e-14
    - Tidal BNS (NRTidalv2, NRTidalv3, TaylorF2): mismatch < 1e-13
    - Precessing (IMRPhenomPv2): mismatch < 1e-11
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from ripplegw import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.constants import PI
from tests.utils import (
    check_lal_available,
    check_is_tidal,
    check_is_precessing,
    get_freqs,
    get_jitted_waveform,
    get_lal_waveform,
    get_nyquist_mask,
    compute_match,
    load_psd,
    generate_random_params,
    LAL_AVAILABLE,
)

jax.config.update("jax_enable_x64", True)

# Skip all tests if LALSuite is not available
pytestmark = pytest.mark.skipif(
    not LAL_AVAILABLE,
    reason="LALSuite required for cross-validation tests",
)


# ============================================================================
# Test configuration
# ============================================================================

# Default parameter bounds for random sampling
DEFAULT_BOUNDS = {
    "m": [0.5, 3.0],  # Masses (solar masses) - BNS range
    "chi": [0.0, 0.05],  # Spins (low for BNS)
    "lambda": [0.0, 5000.0],  # Tidal parameters
    "d_L": [30.0, 300.0],  # Distance (Mpc)
}

BBH_BOUNDS = {
    "m": [5.0, 50.0],  # Masses (solar masses) - BBH range
    "chi": [-0.99, 0.99],  # Spins
    "lambda": [0.0, 0.0],  # No tidal
    "d_L": [100.0, 1000.0],  # Distance (Mpc)
}

# Number of random samples to test (can be overridden with pytest --count=N)
N_SAMPLES = 200

# Mismatch thresholds for each waveform type
MISMATCH_THRESHOLDS = {
    "IMRPhenomD": 1e-14,
    "IMRPhenomXAS": 1e-14,
    "IMRPhenomD_NRTidalv2": 1e-13,
    "IMRPhenomXAS_NRTidalv3": 1e-13,
    "TaylorF2": 1e-13,
    "IMRPhenomPv2": 1e-11,
}


# ============================================================================
# Helper functions
# ============================================================================


def compute_ripple_lal_mismatch(
    waveform_name: str,
    theta_lal: np.ndarray,
    fs: jnp.ndarray,
    f_l: float,
    f_u: float,
    df: float,
    f_ref: float,
    psd: np.ndarray,
    psd_freqs: np.ndarray,
) -> float:
    """Compute mismatch between ripple and LAL waveforms.

    Args:
        waveform_name: Name of the waveform.
        theta_lal: Parameter array in LAL format.
        fs: Ripple frequency array.
        f_l: Lower frequency.
        f_u: Upper frequency.
        df: Frequency spacing.
        f_ref: Reference frequency.
        psd: PSD values.
        psd_freqs: PSD frequencies.

    Returns:
        Mismatch (1 - match).
    """
    is_tidal = check_is_tidal(waveform_name)
    is_precessing = check_is_precessing(waveform_name)

    # Generate LAL waveform
    hp_lal = get_lal_waveform(
        theta_lal, waveform_name, f_l, f_u, df, f_ref, is_tidal, is_precessing
    )

    # Convert parameters to ripple format
    if is_precessing:
        # Precessing: theta_lal = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist, tc, phic, inc]
        # Ripple IMRPhenomPv2 expects: [Mc, eta, chi1_l, chi2_l, chi1, chi2, theta1, theta2, phi_12, phi_jl, dist, tc, phic, inc]
        # This is complex - for now we'll skip precessing in this implementation
        # TODO: Implement proper precessing parameter conversion
        raise NotImplementedError(
            "Precessing waveform cross-validation not yet implemented"
        )
    else:
        m1 = theta_lal[0]
        m2 = theta_lal[1]
        Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

        if is_tidal:
            s1z = theta_lal[2]
            s2z = theta_lal[3]
            l1 = theta_lal[4]
            l2 = theta_lal[5]
            lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
                jnp.array([l1, l2, m1, m2])
            )
            dist_mpc = theta_lal[6]
            tc = theta_lal[7]
            phic = theta_lal[8]
            inclination = theta_lal[9]
            theta_ripple = jnp.array(
                [
                    Mc,
                    eta,
                    s1z,
                    s2z,
                    lambda_tilde,
                    delta_lambda_tilde,
                    dist_mpc,
                    tc,
                    phic,
                    inclination,
                ]
            )
        else:
            s1z = theta_lal[2]
            s2z = theta_lal[3]
            dist_mpc = theta_lal[4]
            tc = theta_lal[5]
            phic = theta_lal[6]
            inclination = theta_lal[7]
            theta_ripple = jnp.array(
                [Mc, eta, s1z, s2z, dist_mpc, tc, phic, inclination]
            )

    # Generate ripple waveform
    waveform = get_jitted_waveform(waveform_name, fs, f_ref)
    hp_ripple = waveform(theta_ripple)

    # Apply Nyquist mask to both waveforms
    nyquist_mask = get_nyquist_mask(fs)
    hp_lal_masked = jnp.array(hp_lal) * nyquist_mask
    hp_ripple_masked = hp_ripple * nyquist_mask

    # Interpolate PSD to ripple frequency grid
    psd_interp = jnp.interp(fs, psd_freqs, psd)

    # Compute match
    match = compute_match(hp_ripple_masked, hp_lal_masked, psd_interp, fs)
    mismatch = 1.0 - match

    return mismatch


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def freq_params():
    """Default frequency parameters."""
    return {
        "f_l": 20.0,
        "f_u": 1024.0,
        "f_sampling": 2048.0,
        "T": 256.0,
        "f_ref": 20.0,
    }


@pytest.fixture
def psd_data():
    """Load PSD for cross-validation."""
    return load_psd("psd.txt")


# ============================================================================
# Parametrized tests for all waveforms
# ============================================================================


@pytest.mark.parametrize(
    "waveform_name,bounds,n_samples",
    [
        ("IMRPhenomD", BBH_BOUNDS, N_SAMPLES),
        ("IMRPhenomXAS", BBH_BOUNDS, N_SAMPLES),
        ("IMRPhenomD_NRTidalv2", DEFAULT_BOUNDS, N_SAMPLES),
        ("IMRPhenomXAS_NRTidalv3", DEFAULT_BOUNDS, N_SAMPLES),
        ("TaylorF2", DEFAULT_BOUNDS, N_SAMPLES),
        # IMRPhenomPv2 requires special handling - skip for now
        # ("IMRPhenomPv2", BBH_BOUNDS, N_SAMPLES),
    ],
)
def test_waveform_mismatch(
    waveform_name, bounds, n_samples, freq_params, psd_data
):
    """Test that ripple waveforms match LALSuite to machine precision.

    This test generates random parameter sets, computes both LAL and ripple
    waveforms, and verifies that the mismatch is below the threshold.
    """
    check_lal_available()

    # Unpack parameters
    f_l = freq_params["f_l"]
    f_u = freq_params["f_u"]
    f_sampling = freq_params["f_sampling"]
    T = freq_params["T"]
    f_ref = freq_params["f_ref"]
    psd_freqs, psd = psd_data

    # Construct frequency grid
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = fs[1] - fs[0]

    # Generate random parameters
    is_tidal = check_is_tidal(waveform_name)
    is_precessing = check_is_precessing(waveform_name)
    theta_batch = generate_random_params(
        n_samples, bounds, is_tidal=is_tidal, is_precessing=is_precessing, seed=42
    )

    # Compute mismatches for all samples
    mismatches = []
    failed_params = []

    for i, theta_lal in enumerate(theta_batch):
        try:
            mismatch = compute_ripple_lal_mismatch(
                waveform_name, theta_lal, fs, f_l, f_u, df, f_ref, psd, psd_freqs
            )
            mismatches.append(mismatch)

            # Check NaN/Inf
            if not np.isfinite(mismatch):
                failed_params.append((i, theta_lal, "NaN/Inf mismatch"))

        except Exception as e:
            failed_params.append((i, theta_lal, str(e)))
            mismatches.append(np.nan)

    mismatches = np.array(mismatches)
    finite_mismatches = mismatches[np.isfinite(mismatches)]

    # Save results to CSV
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"mismatch_{waveform_name}.csv"

    # Build dataframe
    if is_tidal:
        df_data = {
            "m1": theta_batch[:, 0],
            "m2": theta_batch[:, 1],
            "chi1": theta_batch[:, 2],
            "chi2": theta_batch[:, 3],
            "lambda1": theta_batch[:, 4],
            "lambda2": theta_batch[:, 5],
            "dist_mpc": theta_batch[:, 6],
            "tc": theta_batch[:, 7],
            "phi_ref": theta_batch[:, 8],
            "inclination": theta_batch[:, 9],
            "mismatch": mismatches,
            "log10_mismatch": np.log10(mismatches),
        }
    else:
        df_data = {
            "m1": theta_batch[:, 0],
            "m2": theta_batch[:, 1],
            "chi1": theta_batch[:, 2],
            "chi2": theta_batch[:, 3],
            "dist_mpc": theta_batch[:, 4],
            "tc": theta_batch[:, 5],
            "phi_ref": theta_batch[:, 6],
            "inclination": theta_batch[:, 7],
            "mismatch": mismatches,
            "log10_mismatch": np.log10(mismatches),
        }

    df = pd.DataFrame(df_data)
    df = df.sort_values(by="mismatch", ascending=False)
    df.to_csv(results_file, index=False)

    # Print statistics
    print(f"\n{waveform_name} Mismatch Statistics:")
    print(f"  Samples: {n_samples}")
    print(f"  Mean mismatch: {np.mean(finite_mismatches):.2e}")
    print(f"  Median mismatch: {np.median(finite_mismatches):.2e}")
    print(f"  Min mismatch: {np.min(finite_mismatches):.2e}")
    print(f"  Max mismatch: {np.max(finite_mismatches):.2e}")
    print(f"  Failed samples: {len(failed_params)}")
    print(f"  Results saved to: {results_file}")

    # Assert that all mismatches are below threshold
    threshold = MISMATCH_THRESHOLDS[waveform_name]
    max_mismatch = np.max(finite_mismatches)

    if failed_params:
        print("\nFailed parameters:")
        for i, theta, error in failed_params[:5]:  # Print first 5 failures
            print(f"  Sample {i}: {error}")
            print(f"    Params: {theta}")

    assert (
        len(failed_params) == 0
    ), f"{len(failed_params)}/{n_samples} samples failed"
    assert (
        max_mismatch < threshold
    ), f"Max mismatch {max_mismatch:.2e} exceeds threshold {threshold:.2e}"


# ============================================================================
# Individual waveform tests for focused debugging
# ============================================================================


def test_imrphenomd_single_point(freq_params, psd_data):
    """Test IMRPhenomD with a single known-good parameter set."""
    check_lal_available()

    waveform_name = "IMRPhenomD"
    f_l = freq_params["f_l"]
    f_u = freq_params["f_u"]
    f_sampling = freq_params["f_sampling"]
    T = freq_params["T"]
    f_ref = freq_params["f_ref"]
    psd_freqs, psd = psd_data

    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = fs[1] - fs[0]

    # Known-good parameters
    theta_lal = np.array([30.0, 25.0, 0.5, -0.3, 400.0, 0.0, 0.5, 0.8])

    mismatch = compute_ripple_lal_mismatch(
        waveform_name, theta_lal, fs, f_l, f_u, df, f_ref, psd, psd_freqs
    )

    print(f"\nIMRPhenomD single point mismatch: {mismatch:.2e}")
    assert mismatch < 1e-14, f"Mismatch {mismatch:.2e} too large"


def test_imrphenomxas_single_point(freq_params, psd_data):
    """Test IMRPhenomXAS with a single known-good parameter set."""
    check_lal_available()

    waveform_name = "IMRPhenomXAS"
    f_l = freq_params["f_l"]
    f_u = freq_params["f_u"]
    f_sampling = freq_params["f_sampling"]
    T = freq_params["T"]
    f_ref = freq_params["f_ref"]
    psd_freqs, psd = psd_data

    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = fs[1] - fs[0]

    # Known-good parameters
    theta_lal = np.array([30.0, 25.0, 0.5, -0.3, 400.0, 0.0, 0.5, 0.8])

    mismatch = compute_ripple_lal_mismatch(
        waveform_name, theta_lal, fs, f_l, f_u, df, f_ref, psd, psd_freqs
    )

    print(f"\nIMRPhenomXAS single point mismatch: {mismatch:.2e}")
    assert mismatch < 1e-14, f"Mismatch {mismatch:.2e} too large"
