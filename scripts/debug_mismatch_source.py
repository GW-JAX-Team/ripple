#!/usr/bin/env python
"""Check if the overlap loss is due to PSD weighting or intrinsic waveform differences."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from tests.utils import (
    get_freqs,
    get_jitted_waveform,
    get_lal_waveform,
    compute_overlap,
    compute_overlap_loss,
    noise_weighted_inner_product,
)

jax.config.update("jax_enable_x64", True)


def main():
    m1, m2 = 1.4, 1.3
    chi1, chi2 = 0.02, -0.01
    lambda1, lambda2 = 400.0, 300.0
    dist_mpc = 100.0
    tc = 0.0
    phic = 0.0
    inclination = np.pi / 4
    
    theta_lal = np.array([m1, m2, chi1, chi2, lambda1, lambda2, dist_mpc, tc, phic, inclination])
    
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
        jnp.array([lambda1, lambda2, m1, m2])
    )
    theta_ripple = jnp.array([
        Mc, eta, chi1, chi2, lambda_tilde, delta_lambda_tilde,
        dist_mpc, tc, phic, inclination
    ])
    
    f_l, f_u, f_sampling, T, f_ref = 20.0, 4096.0, 8192.0, 128.0, 20.0
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = float(fs[1] - fs[0])
    fs_np = np.array(fs)
    
    print(f"Frequency grid:")
    print(f"  df = {df:.10f} Hz")
    print(f"  n_freqs = {len(fs)}")
    print(f"  f[0] = {float(fs[0]):.10f} Hz")
    print(f"  f[1] = {float(fs[1]):.10f} Hz")
    print(f"  f[-1] = {float(fs[-1]):.10f} Hz")
    print()
    
    # Generate waveforms
    hp_lal, hc_lal = get_lal_waveform(
        theta_lal, "IMRPhenomD_NRTidalv2",
        f_l, f_u, df, f_ref,
        is_tidal=True, is_precessing=False
    )
    
    waveform = get_jitted_waveform("IMRPhenomD_NRTidalv2", fs, f_ref)
    hp_ripple, hc_ripple = waveform(theta_ripple)
    hp_ripple_np = np.array(hp_ripple)
    
    # PSD
    psd_path = Path(__file__).parent.parent / "tests" / "psds" / "ET_D_psd.txt"
    psd_freqs_np, psd_np = np.loadtxt(psd_path, unpack=True)
    psd_interp = jnp.interp(fs, jnp.array(psd_freqs_np), jnp.array(psd_np))
    
    # Compute the integrand for the inner product
    from jax.scipy.integrate import trapezoid
    
    h1 = hp_ripple
    h2 = jnp.array(hp_lal)
    
    # Noise-weighted inner products
    h1_sq = noise_weighted_inner_product(h1, h1, psd_interp, fs)
    h2_sq = noise_weighted_inner_product(h2, h2, psd_interp, fs)
    h1_h2_complex = trapezoid(jnp.conj(h1) * h2 / psd_interp, x=fs, axis=-1)
    h1_h2 = 4 * h1_h2_complex
    
    print(f"Inner products:")
    print(f"  <h1|h1> = {float(h1_sq):.15e}")
    print(f"  <h2|h2> = {float(h2_sq):.15e}")
    print(f"  <h1|h2> = {float(h1_h2.real):.15e} + {float(h1_h2.imag):.15e}j")
    print()
    
    # Overlap
    overlap = h1_h2.real / jnp.sqrt(h1_sq * h2_sq)
    print(f"Overlap: {float(overlap):.15e}")
    print(f"Overlap loss: {1.0 - float(overlap):.15e}")
    print()
    
    # Now let's see where the mismatch comes from
    # Compute |h1 - h2|^2 weighted by PSD
    diff = h1 - h2
    diff_sq = noise_weighted_inner_product(diff, diff, psd_interp, fs)
    
    print(f"<h1-h2|h1-h2> = {float(diff_sq):.15e}")
    print(f"sqrt(<h1-h2|h1-h2>) = {float(jnp.sqrt(diff_sq)):.15e}")
    print()
    
    # Check relative amplitude difference
    rel_amp_diff = jnp.abs(h1) / jnp.abs(h2) - 1.0
    print(f"Relative amplitude difference statistics:")
    non_zero_mask = jnp.abs(h2) > 1e-50
    print(f"  Mean: {float(jnp.mean(rel_amp_diff[non_zero_mask])):.15e}")
    print(f"  Max:  {float(jnp.max(jnp.abs(rel_amp_diff[non_zero_mask]))):.15e}")
    print()
    
    # Check phase difference
    phase_diff = jnp.angle(h1) - jnp.angle(h2)
    # Unwrap phase difference
    phase_diff_unwrapped = jnp.unwrap(phase_diff)
    print(f"Phase difference statistics (unwrapped):")
    print(f"  Mean: {float(jnp.mean(phase_diff_unwrapped[non_zero_mask])):.15e}")
    print(f"  Max:  {float(jnp.max(jnp.abs(phase_diff_unwrapped[non_zero_mask]))):.15e}")
    print(f"  Std:  {float(jnp.std(phase_diff_unwrapped[non_zero_mask])):.15e}")
    print()
    
    # The overlap loss formula is:
    # 1 - overlap = (A*B - C^2) / (sqrt(A*B) * (sqrt(A*B) + C))
    # where A = <h1|h1>, B = <h2|h2>, C = Re(<h1|h2>)
    #
    # For small mismatches:
    # 1 - overlap ≈ <h1-h2|h1-h2> / (2 * <h1|h1>)
    #
    # Let me check this approximation
    
    approx_loss = float(diff_sq) / (2.0 * float(h1_sq))
    print(f"Approximate overlap loss (diff_sq / 2*h1_sq): {approx_loss:.15e}")
    print(f"Actual overlap loss: {1.0 - float(overlap):.15e}")
    print()
    
    # The phase difference is the dominant source of mismatch
    # Let me check what phase difference would give this overlap loss
    # For pure phase mismatch: overlap ≈ cos(delta_phi)
    # So delta_phi ≈ arccos(overlap)
    
    import math
    delta_phi = math.acos(float(overlap))
    print(f"Equivalent pure phase mismatch: {delta_phi:.15e} rad")
    print(f"  = {np.degrees(delta_phi):.15e} degrees")
    print()
    
    # This is the RMS phase difference weighted by the PSD
    # Let me check if this matches the actual phase differences
    
    # Compute PSD-weighted phase variance
    weights = jnp.abs(h1) * jnp.abs(h2) / psd_interp
    weights_norm = weights / trapezoid(weights, x=fs)
    phase_var = trapezoid(phase_diff_unwrapped**2 * weights_norm, x=fs)
    phase_rms = float(jnp.sqrt(phase_var))
    
    print(f"PSD-weighted RMS phase difference: {phase_rms:.15e} rad")
    print(f"Expected from overlap: {delta_phi:.15e} rad")


if __name__ == "__main__":
    main()
