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
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from tests.utils import (
    check_lal_available,
    check_is_tidal,
    check_is_precessing,
    get_freqs,
    get_jitted_waveform,
    get_lal_waveform,
    get_nyquist_mask,
    compute_match,
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
    "chi": [-0.05, 0.05],  # Aligned spins (BNS physical range; both signs tested)
    "lambda": [0.0, 5000.0],  # Tidal parameters
    "d_L": [30.0, 500.0],  # Distance (Mpc)
}

BBH_BOUNDS = {
    "m": [5.0, 100.0],  # Masses (solar masses) - BBH range
    "chi": [-0.99, 0.99],  # Spins
    "lambda": [0.0, 0.0],  # No tidal
    "d_L": [100.0, 3000.0],  # Distance (Mpc)
}

# Maximum number of random samples when running the full suite
N_SAMPLES_FULL = 10

# Per-waveform mismatch thresholds.
# These represent expected float64 agreement between the ripple and LAL
# implementations; simpler/more-analytical waveforms achieve near-machine
# precision while complex NR-calibrated ones accumulate more rounding error.
MISMATCH_THRESHOLDS = {
    "IMRPhenomD": 1e-5,         # observed max ~1e-7 over 10 samples
    "IMRPhenomXAS": 1e-12,      # near machine precision (~3e-14)
    "IMRPhenomD_NRTidalv2": 1e-9,   # observed max ~6e-11
    "IMRPhenomXAS_NRTidalv3": 1e-7,  # observed max ~6e-9
    "TaylorF2": 1e-14,          # at float64 machine epsilon (~3e-16)
    "IMRPhenomPv2": 1e-4,       # observed max ~2e-6
    "IMRPhenomXPHM": 1e-3,      # active development (PR #95); expected to exceed
}
DEFAULT_MISMATCH_THRESHOLD = 1e-5  # fallback for unknown waveforms


def get_mismatch_threshold(waveform_name: str) -> float:
    """Return the per-waveform mismatch threshold."""
    return MISMATCH_THRESHOLDS.get(waveform_name, DEFAULT_MISMATCH_THRESHOLD)


# ============================================================================
# Helper functions
# ============================================================================

def convert_parameters_lal_to_ripple(theta_lal: np.ndarray, is_precessing: bool, is_tidal: bool):
    # Convert parameters to ripple format
    if is_precessing:
        # Precessing: theta_lal = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist, tc, phic, inc]
        # Ripple precessing waveforms (IMRPhenomPv2, IMRPhenomXPHM) expect:
        #   [Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phiRef, incl]
        m1 = theta_lal[0]
        m2 = theta_lal[1]
        Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
        s1x = theta_lal[2]
        s1y = theta_lal[3]
        s1z = theta_lal[4]
        s2x = theta_lal[5]
        s2y = theta_lal[6]
        s2z = theta_lal[7]
        dist_mpc = theta_lal[8]
        tc = theta_lal[9]
        phic = theta_lal[10]
        inclination = theta_lal[11]
        theta_ripple = jnp.array(
            [Mc, eta, s1x, s1y, s1z, s2x, s2y, s2z, dist_mpc, tc, phic, inclination]
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
    return theta_ripple


def compute_ripple_lal_mismatch(
    hphc_lal: tuple,
    hphc_ripple: tuple,
    fs: jnp.ndarray,
    f_l: float,
    f_u: float,
    df: float,
    f_ref: float,
    psd: np.ndarray,
    psd_freqs: np.ndarray,
) -> tuple[float, float]:
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
        Tuple (mismatch_hp, mismatch_hc) where each is 1 - match for the
        respective polarization. For aligned-spin waveforms hc = i*hp in the
        frequency domain, so both should agree. For precessing waveforms the
        two polarizations are independent and both are tested.
    """
    hp_lal, hc_lal = hphc_lal
    hp_ripple, hc_ripple = hphc_ripple
    
    # Apply Nyquist mask to all waveforms
    nyquist_mask = get_nyquist_mask(fs)
    hp_lal_masked = jnp.array(hp_lal) * nyquist_mask
    hc_lal_masked = jnp.array(hc_lal) * nyquist_mask
    hp_ripple_masked = hp_ripple * nyquist_mask
    hc_ripple_masked = hc_ripple * nyquist_mask

    # Interpolate PSD to ripple frequency grid
    psd_interp = jnp.interp(fs, psd_freqs, psd)

    # Compute match for both polarizations
    match_hp = compute_match(hp_ripple_masked, hp_lal_masked, psd_interp, fs)
    match_hc = compute_match(hc_ripple_masked, hc_lal_masked, psd_interp, fs)
    mismatch_hp = 1.0 - match_hp
    mismatch_hc = 1.0 - match_hc

    return mismatch_hp, mismatch_hc


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
    """Load aLIGO PSD for cross-validation.

    Source: bilby (https://github.com/bilby-dev/bilby)
    Commit: 0985f75c664786e21cc4f662d4f12fe181b1a536
    """
    psd_path = Path(__file__).parent.parent / "psds" / "ET_D_psd.txt"
    freqs, psd = np.loadtxt(psd_path, unpack=True)
    return freqs, psd


# ============================================================================
# Parametrized tests for all waveforms
# ============================================================================


@pytest.mark.parametrize(
    "waveform_name,bounds",
    [
        # ("IMRPhenomD", BBH_BOUNDS),
        # ("IMRPhenomXAS", BBH_BOUNDS),
        # ("IMRPhenomD_NRTidalv2", DEFAULT_BOUNDS),
        # ("IMRPhenomXAS_NRTidalv3", DEFAULT_BOUNDS),
        # ("TaylorF2", DEFAULT_BOUNDS),
        # ("IMRPhenomPv2", BBH_BOUNDS),
        ("IMRPhenomXPHM", BBH_BOUNDS),
    ],
)
def test_waveform_mismatch(waveform_name, bounds, freq_params, cross_val_results, psd_data):
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
        N_SAMPLES_FULL, bounds, is_tidal=is_tidal, is_precessing=is_precessing, seed=42
    )

    # Compute mismatches for all samples
    mismatches_hp = []
    mismatches_hc = []
    failed_params = []

    is_tidal = check_is_tidal(waveform_name)
    is_precessing = check_is_precessing(waveform_name)

    # Generate ripple waveform 
    waveform = get_jitted_waveform(waveform_name, fs, f_ref)

    for i, theta_lal in enumerate(theta_batch):
        try:
            # Generate LAL waveform (both polarizations)
            hphc_lal = get_lal_waveform(
                theta_lal, waveform_name, f_l, f_u, df, f_ref, is_tidal, is_precessing
            )
            # Get ripple parameters
            theta_ripple = convert_parameters_lal_to_ripple(theta_lal, is_precessing, is_tidal)
            # Generate ripple waveform
            hphc_ripple = waveform(theta_ripple)
            mismatch_hp, mismatch_hc = compute_ripple_lal_mismatch(
                hphc_lal, hphc_ripple, fs, f_l, f_u, df, f_ref, psd, psd_freqs
            )
            mismatches_hp.append(mismatch_hp)
            mismatches_hc.append(mismatch_hc)

            # Check NaN/Inf
            if not np.isfinite(mismatch_hp) or not np.isfinite(mismatch_hc):
                failed_params.append((i, theta_lal, "NaN/Inf mismatch"))

        except Exception as e:
            failed_params.append((i, theta_lal, str(e)))
            mismatches_hp.append(np.nan)
            mismatches_hc.append(np.nan)

    mismatches_hp = np.array(mismatches_hp)
    mismatches_hc = np.array(mismatches_hc)
    # Worst-case mismatch over both polarizations
    mismatches = np.maximum(mismatches_hp, mismatches_hc)
    finite_mismatches = mismatches[np.isfinite(mismatches)]

    n_nonfinite = mismatches.size - finite_mismatches.size
    assert finite_mismatches.size > 0, (
        f"All {mismatches.size} per-sample mismatches are non-finite for {waveform_name}. "
        f"Non-finite count: {n_nonfinite}. "
        f"Sample mismatches (first 10): {mismatches[:10]}. "
        f"Failed samples: {len(failed_params)}."
    )

    # Save results to CSV
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"mismatch_{waveform_name}.csv"

    # Build dataframe
    if is_tidal:
        m1, m2 = theta_batch[:, 0], theta_batch[:, 1]
        chi1z, chi2z = theta_batch[:, 2], theta_batch[:, 3]
        lambda1, lambda2 = theta_batch[:, 4], theta_batch[:, 5]
        df_data = {
            "m1": m1,
            "m2": m2,
            "chi1z": chi1z,
            "chi2z": chi2z,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "dist_mpc": theta_batch[:, 6],
            "tc": theta_batch[:, 7],
            "phi_ref": theta_batch[:, 8],
            "inclination": theta_batch[:, 9],
            "mismatch_hp": mismatches_hp,
            "mismatch_hc": mismatches_hc,
            "mismatch": mismatches,
        }
    elif is_precessing:
        m1, m2 = theta_batch[:, 0], theta_batch[:, 1]
        s1x, s1y, s1z = theta_batch[:, 2], theta_batch[:, 3], theta_batch[:, 4]
        s2x, s2y, s2z = theta_batch[:, 5], theta_batch[:, 6], theta_batch[:, 7]
        chi1z, chi2z = s1z, s2z
        df_data = {
            "m1": m1,
            "m2": m2,
            "s1x": s1x,
            "s1y": s1y,
            "s1z": s1z,
            "s2x": s2x,
            "s2y": s2y,
            "s2z": s2z,
            "chi1_mag": np.sqrt(s1x**2 + s1y**2 + s1z**2),
            "chi2_mag": np.sqrt(s2x**2 + s2y**2 + s2z**2),
            "dist_mpc": theta_batch[:, 8],
            "tc": theta_batch[:, 9],
            "phi_ref": theta_batch[:, 10],
            "inclination": theta_batch[:, 11],
            "mismatch_hp": mismatches_hp,
            "mismatch_hc": mismatches_hc,
            "mismatch": mismatches,
        }
    else:
        m1, m2 = theta_batch[:, 0], theta_batch[:, 1]
        chi1z, chi2z = theta_batch[:, 2], theta_batch[:, 3]
        df_data = {
            "m1": m1,
            "m2": m2,
            "chi1z": chi1z,
            "chi2z": chi2z,
            "dist_mpc": theta_batch[:, 4],
            "tc": theta_batch[:, 5],
            "phi_ref": theta_batch[:, 6],
            "inclination": theta_batch[:, 7],
            "mismatch_hp": mismatches_hp,
            "mismatch_hc": mismatches_hc,
            "mismatch": mismatches,
        }

    df = pd.DataFrame(df_data)
    # Derived columns
    df["m_total"] = df["m1"] + df["m2"]
    df["mass_ratio"] = np.minimum(df["m1"], df["m2"]) / np.maximum(df["m1"], df["m2"])
    if is_precessing:
        df["chi_eff"] = (df["m1"] * df["s1z"] + df["m2"] * df["s2z"]) / df["m_total"]
    else:
        df["chi_eff"] = (df["m1"] * df["chi1z"] + df["m2"] * df["chi2z"]) / df[
            "m_total"
        ]
    df["log10_mismatch"] = np.log10(np.abs(df["mismatch"].clip(1e-30)))
    df = df.sort_values(by="mismatch", ascending=False)
    df.to_csv(results_file, index=False)

    # Print statistics
    print(f"\n{waveform_name} Mismatch Statistics:")
    print(f"  Samples: {N_SAMPLES_FULL}")
    print(f"  Mean mismatch: {np.mean(finite_mismatches):.2e}")
    print(f"  Median mismatch: {np.median(finite_mismatches):.2e}")
    print(f"  Min mismatch: {np.min(finite_mismatches):.2e}")
    print(f"  Max mismatch: {np.max(finite_mismatches):.2e}")
    print(f"  Failed samples: {len(failed_params)}")
    print(f"  Results saved to: {results_file}")

    # Plot mismatch distribution
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    mismatch_threshold = get_mismatch_threshold(waveform_name)
    log10_thresh = np.log10(mismatch_threshold)
    log10_m = df["log10_mismatch"].values
    finite_mask = np.isfinite(log10_m)
    log10_m_finite = log10_m[finite_mask]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"{waveform_name}: ripple vs LAL mismatch", fontsize=13)

    # (0,0) - histogram
    ax = axes[0, 0]
    ax.hist(log10_m_finite, bins=30, edgecolor="black", alpha=0.8)
    ax.axvline(
        log10_thresh,
        color="red",
        linestyle="--",
        label=f"threshold = {mismatch_threshold:.0e}",
    )
    ax.set_xlabel(r"$\log_{10}$(mismatch)")
    ax.set_ylabel("Count")
    ax.set_title("Mismatch distribution")
    ax.legend()

    # (0,1) - mass ratio vs total mass colored by mismatch
    ax = axes[0, 1]
    sc = ax.scatter(
        df["m_total"][finite_mask],
        df["mass_ratio"][finite_mask],
        c=log10_m_finite,
        cmap="viridis",
        s=20,
        alpha=0.8,
    )
    ax.set_xlabel(r"$M_{\rm total}\;[M_\odot]$")
    ax.set_ylabel(r"$q = m_2/m_1$")
    ax.set_title(r"Mass plane (colour = $\log_{10}$ mismatch)")
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}$(mismatch)")

    # (1,0) - mismatch vs chi_eff / lambda_tilde / chi_mag
    ax = axes[1, 0]
    if is_tidal:
        # Compute lambda_tilde from lambda1, lambda2, m1, m2
        eta = df["m1"] * df["m2"] / df["m_total"] ** 2
        x_vals = (8.0 / 13.0) * (
            (1 + 7 * eta - 31 * eta**2) * (df["lambda1"] + df["lambda2"])
            + (1 - 4 * eta) ** 0.5
            * (1 + 9 * eta - 11 * eta**2)
            * (df["lambda1"] - df["lambda2"])
        )
        x_label = r"$\tilde{\Lambda}$"
    elif is_precessing:
        x_vals = df["chi1_mag"]
        x_label = r"$|\chi_1|$"
    else:
        x_vals = df["chi_eff"]
        x_label = r"$\chi_{\rm eff}$"
    sc = ax.scatter(
        x_vals[finite_mask],
        df["inclination"][finite_mask],
        c=log10_m_finite,
        cmap="viridis",
        s=20,
        alpha=0.8,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Inclination [rad]")
    ax.set_title(f"{x_label} vs inclination (colour = $\log_{{10}}$ mismatch)")
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}$(mismatch)")

    # (1,1) - 2D: m1 vs m2 colored by mismatch
    ax = axes[1, 1]
    sc = ax.scatter(
        df["m1"][finite_mask],
        df["m2"][finite_mask],
        c=log10_m_finite,
        cmap="plasma",
        s=30,
        alpha=0.9,
    )
    ax.set_xlabel(r"$m_1\;[M_\odot]$")
    ax.set_ylabel(r"$m_2\;[M_\odot]$")
    ax.set_title(r"$m_1$ vs $m_2$ (colour = $\log_{10}$ mismatch)")
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}$(mismatch)")

    fig.tight_layout()
    fig_file = figures_dir / f"mismatch_{waveform_name}.png"
    fig.savefig(fig_file, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to: {fig_file}")

    # Assert that all mismatches are below threshold
    max_mismatch = np.max(finite_mismatches)
    mismatch_threshold = get_mismatch_threshold(waveform_name)

    if failed_params:
        print("\nFailed parameters:")
        for i, theta, error in failed_params[:5]:  # Print first 5 failures
            print(f"  Sample {i}: {error}")
            print(f"    Params: {theta}")

    # Record stats for the session-level summary
    cross_val_results.append(
        {
            "waveform": waveform_name,
            "n_samples": N_SAMPLES_FULL,
            "n_finite": len(finite_mismatches),
            "n_failed": len(failed_params),
            "mean": float(np.mean(finite_mismatches)),
            "median": float(np.median(finite_mismatches)),
            "min": float(np.min(finite_mismatches)),
            "max": float(max_mismatch),
            "threshold": mismatch_threshold,
            "passed": bool(
                len(failed_params) == 0 and max_mismatch < mismatch_threshold
            ),
        }
    )

    assert len(failed_params) == 0, f"{len(failed_params)}/{N_SAMPLES_FULL} samples failed"
    assert max_mismatch < mismatch_threshold, (
        f"Max mismatch {max_mismatch:.2e} exceeds threshold {mismatch_threshold:.2e}"
    )
