#!/usr/bin/env python
# ruff: noqa: E402
"""Baseline benchmark for IMRPhenomD_NRTidalv2 overlap loss between LALSuite and Ripple.

This script generates N random parameter sets, computes both LAL and Ripple
waveforms, and calculates the noise-weighted overlap loss for each. Results
and parameters are saved to CSV for reproducibility.

Usage:
    uv run scripts/baseline_imrphenomd_nrtidalv2.py [--n-samples N] [--output-dir PATH]

The PSD file used is tests/psds/ET_D_psd.txt (Einstein Telescope D configuration).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.conversions import lambdas_to_lambda_tildes, ms_to_Mc_eta
from tests.utils import (
    LAL_AVAILABLE,
    compute_overlap_loss,
    generate_random_params,
    get_freqs,
    get_jitted_waveform,
    get_lal_waveform,
    get_nyquist_mask,
)

# Default parameter bounds for BNS range
DEFAULT_BOUNDS = {
    "m": [0.5, 3.0],  # Masses (solar masses) - BNS range
    "chi": [-0.05, 0.05],  # Aligned spins (BNS physical range; both signs tested)
    "lambda": [0.0, 5000.0],  # Tidal parameters
    "d_L": [30.0, 500.0],  # Distance (Mpc)
}


def convert_parameters_lal_to_ripple(theta_lal: np.ndarray) -> jnp.ndarray:
    """Convert LAL parameters to Ripple format for IMRPhenomD_NRTidalv2.

    LAL format: [m1, m2, s1z, s2z, l1, l2, dist, tc, phic, inc]
    Ripple format: [Mc, eta, chi1z, chi2z, lambda_tilde, delta_lambda_tilde, dist_mpc, tc, phic, inclination]
    """
    m1 = theta_lal[0]
    m2 = theta_lal[1]
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

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

    return jnp.array(
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


def main():
    parser = argparse.ArgumentParser(
        description="Baseline benchmark for IMRPhenomD_NRTidalv2 overlap loss"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of random parameter sets to generate (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_results/imrphenomd_nrtidalv2_baseline",
        help="Output directory for results (default: baseline_results/imrphenomd_nrtidalv2_baseline)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    if args.n_samples <= 0:
        parser.error("--n-samples must be a positive integer")

    if not LAL_AVAILABLE:
        print("ERROR: LALSuite is required for this benchmark.")
        print("Install it with: uv sync --all-extras --dev")
        sys.exit(1)

    # Configuration
    n_samples = args.n_samples
    output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Frequency parameters (BNS configuration)
    T = 128.0  # Duration in seconds
    f_l = 20.0  # Lower frequency (Hz)
    f_u = 4096.0  # Upper frequency (Hz)
    f_sampling = 2 * f_u  # Sampling frequency (Hz)
    f_ref = 20.0  # Reference frequency (Hz)

    print("Configuration:")
    print(f"  n_samples = {n_samples}")
    print(f"  T = {T}s, f_l = {f_l} Hz, f_u = {f_u} Hz, f_ref = {f_ref} Hz")
    print(f"  seed = {args.seed}")
    print()

    # Construct frequency grid
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = float(fs[1] - fs[0])
    print(f"  n_freqs = {len(fs)}, df = {df:.4f} Hz")

    # Load PSD
    psd_path = Path(__file__).parent.parent / "tests" / "psds" / "ET_D_psd.txt"
    psd_freqs_np, psd_np = np.loadtxt(psd_path, unpack=True)

    # Generate random parameters
    bounds = DEFAULT_BOUNDS
    theta_lal_batch = generate_random_params(
        n_samples, bounds, is_tidal=True, is_precessing=False, seed=args.seed
    )
    print(f"\nGenerated {n_samples} random parameter sets")

    # Generate LAL waveforms
    print("\nGenerating LAL waveforms...")
    hp_lal_list = []
    hc_lal_list = []
    theta_ripple_list = []
    valid_mask = np.zeros(n_samples, dtype=bool)
    failed_indices = []

    for i in range(n_samples):
        theta_lal = theta_lal_batch[i]
        try:
            hp_lal, hc_lal = get_lal_waveform(
                theta_lal,
                "IMRPhenomD_NRTidalv2",
                f_l,
                f_u,
                df,
                f_ref,
                is_tidal=True,
                is_precessing=False,
            )
            hp_lal_list.append(jnp.array(hp_lal))
            hc_lal_list.append(jnp.array(hc_lal))
            theta_ripple_list.append(convert_parameters_lal_to_ripple(theta_lal))
            valid_mask[i] = True
        except Exception as e:
            failed_indices.append(i)
            print(f"  Sample {i} failed: {e}")

    n_valid = int(valid_mask.sum())
    n_failed = len(failed_indices)
    print(f"  Valid: {n_valid}/{n_samples}")
    print(f"  Failed: {n_failed}/{n_samples}")

    if n_valid == 0:
        print("ERROR: No valid LAL waveforms generated.")
        sys.exit(1)

    # Generate Ripple waveforms and compute overlap loss
    print("\nGenerating Ripple waveforms and computing overlap loss...")
    waveform = get_jitted_waveform("IMRPhenomD_NRTidalv2", fs, f_ref)

    # Prepare batched inputs
    nyquist_mask = get_nyquist_mask(fs)
    psd_interp = jnp.interp(fs, jnp.array(psd_freqs_np), jnp.array(psd_np))

    theta_ripple_batch = jnp.stack(theta_ripple_list)
    hp_lal_batch = jnp.stack(hp_lal_list) * nyquist_mask
    hc_lal_batch = jnp.stack(hc_lal_list) * nyquist_mask

    overlap_loss_hp_np = np.empty(n_valid)
    overlap_loss_hc_np = np.empty(n_valid)
    for i in range(n_valid):
        hp_rip, hc_rip = waveform(theta_ripple_batch[i])
        overlap_loss_hp_np[i] = float(
            compute_overlap_loss(
                hp_rip * nyquist_mask,
                hp_lal_batch[i],
                psd_interp,
                fs,
            )
        )
        overlap_loss_hc_np[i] = float(
            compute_overlap_loss(
                hc_rip * nyquist_mask,
                hc_lal_batch[i],
                psd_interp,
                fs,
            )
        )

    # Assemble results
    overlap_losses_hp = np.full(n_samples, np.nan)
    overlap_losses_hc = np.full(n_samples, np.nan)
    overlap_losses_hp[valid_mask] = overlap_loss_hp_np
    overlap_losses_hc[valid_mask] = overlap_loss_hc_np
    overlap_losses = np.maximum(overlap_losses_hp, overlap_losses_hc)

    # Statistics
    finite_mask = np.isfinite(overlap_losses)
    finite_losses = overlap_losses[finite_mask]
    if finite_losses.size == 0:
        print("ERROR: No finite overlap-loss values were produced.")
        return 1
    finite_losses_for_log = np.clip(finite_losses, 1e-300, 1.0)

    print("\nResults:")
    print(f"  Valid samples: {n_valid}/{n_samples}")
    print(f"  Failed samples: {len(failed_indices)}")
    print("  Overlap loss statistics (worst-case polarization):")
    print(f"    Min:  {finite_losses.min():.4e}")
    print(f"    Max:  {finite_losses.max():.4e}")
    print(f"    Mean: {finite_losses.mean():.4e}")
    print(f"    Median: {np.median(finite_losses):.4e}")
    print(f"    log10(Min):  {np.log10(finite_losses_for_log.min()):.4f}")
    print(f"    log10(Max):  {np.log10(finite_losses_for_log.max()):.4f}")
    print(f"    log10(Mean): {np.log10(finite_losses_for_log.mean()):.4f}")
    print(f"    log10(Median): {np.log10(np.median(finite_losses_for_log)):.4f}")

    # Check threshold
    threshold = 1e-12
    log10_threshold = -12.0
    n_above_threshold = np.sum(finite_losses > threshold)
    print(
        f"\n  Samples with overlap_loss > {threshold:.0e}: {n_above_threshold}/{n_valid}"
    )
    print(f"  Target: log10(overlap_loss) < {log10_threshold}")

    above_threshold_mask = finite_losses > threshold
    if np.any(above_threshold_mask):
        print(f"  WARNING: {n_above_threshold} samples exceed the target threshold!")
        print(
            f"  Worst log10(overlap_loss): {np.log10(finite_losses_for_log.max()):.4f}"
        )
    else:
        print("  OK: All samples meet the target threshold.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"baseline_n{n_samples}_{timestamp}.csv"
    params_file = output_dir / f"baseline_params_n{n_samples}_{timestamp}.csv"
    summary_file = output_dir / f"baseline_summary_{timestamp}.txt"

    # Build DataFrame with results
    m1 = theta_lal_batch[:, 0]
    m2 = theta_lal_batch[:, 1]
    chi1z = theta_lal_batch[:, 2]
    chi2z = theta_lal_batch[:, 3]
    lambda1 = theta_lal_batch[:, 4]
    lambda2 = theta_lal_batch[:, 5]

    df_data = {
        "m1": m1,
        "m2": m2,
        "chi1z": chi1z,
        "chi2z": chi2z,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "dist_mpc": theta_lal_batch[:, 6],
        "tc": theta_lal_batch[:, 7],
        "phi_ref": theta_lal_batch[:, 8],
        "inclination": theta_lal_batch[:, 9],
        "overlap_loss_hp": overlap_losses_hp,
        "overlap_loss_hc": overlap_losses_hc,
        "overlap_loss": overlap_losses,
        "log10_overlap_loss": np.log10(np.clip(overlap_losses, 1e-300, 1.0)),
        "valid": valid_mask,
    }
    df_results = pd.DataFrame(df_data)
    df_results.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Save parameters separately for reproducibility
    df_params = pd.DataFrame(
        {
            "m1": m1,
            "m2": m2,
            "chi1z": chi1z,
            "chi2z": chi2z,
            "lambda1": lambda1,
            "lambda2": lambda2,
            "dist_mpc": theta_lal_batch[:, 6],
            "tc": theta_lal_batch[:, 7],
            "phi_ref": theta_lal_batch[:, 8],
            "inclination": theta_lal_batch[:, 9],
        }
    )
    df_params.to_csv(params_file, index=False)
    print(f"Parameters saved to: {params_file}")

    # Save summary
    with open(summary_file, "w") as f:
        f.write("IMRPhenomD_NRTidalv2 Baseline Benchmark Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"n_samples: {n_samples}\n")
        f.write(f"n_valid: {n_valid}\n")
        f.write(f"n_failed: {len(failed_indices)}\n")
        f.write(f"seed: {args.seed}\n")
        f.write("\nFrequency configuration:\n")
        f.write(f"  T = {T}s\n")
        f.write(f"  f_l = {f_l} Hz\n")
        f.write(f"  f_u = {f_u} Hz\n")
        f.write(f"  f_sampling = {f_sampling} Hz\n")
        f.write(f"  f_ref = {f_ref} Hz\n")
        f.write(f"  n_freqs = {len(fs)}\n")
        f.write(f"  df = {df:.4f} Hz\n")
        f.write(f"\nPSD: {psd_path}\n")
        f.write("\nOverlap loss statistics (worst-case polarization):\n")
        f.write(f"  Min:  {finite_losses.min():.4e}\n")
        f.write(f"  Max:  {finite_losses.max():.4e}\n")
        f.write(f"  Mean: {finite_losses.mean():.4e}\n")
        f.write(f"  Median: {np.median(finite_losses):.4e}\n")
        f.write(f"  log10(Min):  {np.log10(finite_losses_for_log.min()):.4f}\n")
        f.write(f"  log10(Max):  {np.log10(finite_losses_for_log.max()):.4f}\n")
        f.write(f"  log10(Mean): {np.log10(finite_losses_for_log.mean()):.4f}\n")
        f.write(f"  log10(Median): {np.log10(np.median(finite_losses_for_log)):.4f}\n")
        f.write(f"\nThreshold check (overlap_loss < {threshold:.0e}):\n")
        f.write(f"  n_above_threshold: {n_above_threshold}/{n_valid}\n")
        if n_above_threshold > 0:
            f.write(f"  STATUS: FAIL - {n_above_threshold} samples exceed threshold\n")
        else:
            f.write("  STATUS: PASS - All samples meet threshold\n")
        f.write(f"\nFailed sample indices: {failed_indices}\n")

    print(f"Summary saved to: {summary_file}")

    # Return exit code based on threshold check
    return 1 if (n_above_threshold > 0 or n_failed > 0) else 0


if __name__ == "__main__":
    sys.exit(main())
