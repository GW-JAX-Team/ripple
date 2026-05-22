#!/usr/bin/env python
"""Check if the overlap loss is dominated by frequencies near/above NRTIDAL_FMAX."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import _get_merger_frequency, get_kappa
from tests.utils import (
    get_freqs,
    get_jitted_waveform,
    get_lal_waveform,
    get_nyquist_mask,
    compute_overlap_loss,
    generate_random_params,
)

jax.config.update("jax_enable_x64", True)


def main():
    # Use a single test case
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
    
    # Generate waveforms
    hp_lal, hc_lal = get_lal_waveform(
        theta_lal, "IMRPhenomD_NRTidalv2",
        f_l, f_u, df, f_ref,
        is_tidal=True, is_precessing=False
    )
    
    waveform = get_jitted_waveform("IMRPhenomD_NRTidalv2", fs, f_ref)
    hp_ripple, hc_ripple = waveform(theta_ripple)
    hp_ripple_np = np.array(hp_ripple)
    
    # Compute merger frequency
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    f_merger = float(_get_merger_frequency(theta_intrinsic))
    NRTIDAL_FMAX = 1.3 * f_merger
    
    print(f"f_merger = {f_merger:.1f} Hz")
    print(f"NRTIDAL_FMAX = {NRTIDAL_FMAX:.1f} Hz")
    print()
    
    # Check amplitude at different frequencies
    print(f"=== Amplitude comparison ===")
    test_freqs = np.array([100, 500, 1000, 2000, 2500, 2600, 2700, 3000, 4000])
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        amp_lal = np.abs(hp_lal[idx])
        amp_rip = np.abs(hp_ripple_np[idx])
        print(f"  f={tf:.0f} Hz: |hp_LAL| = {amp_lal:.15e}, |hp_Ripple| = {amp_rip:.15e}")
    print()
    
    # Compute overlap loss with different frequency cutoffs
    psd_path = Path(__file__).parent.parent / "tests" / "psds" / "ET_D_psd.txt"
    psd_freqs_np, psd_np = np.loadtxt(psd_path, unpack=True)
    psd_interp = jnp.interp(fs, jnp.array(psd_freqs_np), jnp.array(psd_np))
    
    nyquist_mask = get_nyquist_mask(fs)
    
    # Full overlap loss
    overlap_loss_full = compute_overlap_loss(
        hp_ripple * nyquist_mask, jnp.array(hp_lal) * nyquist_mask,
        psd_interp, fs
    )
    print(f"Full overlap loss (up to {f_u} Hz): {float(overlap_loss_full):.15e}")
    print(f"log10(overlap_loss): {np.log10(float(overlap_loss_full)):.4f}")
    print()
    
    # Overlap loss truncated at NRTIDAL_FMAX
    f_cutoff = NRTIDAL_FMAX
    mask_below = fs_np < f_cutoff
    fs_trunc = fs[mask_below]
    hp_lal_trunc = jnp.array(hp_lal)[mask_below]
    hp_rip_trunc = hp_ripple[mask_below]
    psd_trunc = psd_interp[mask_below]
    nyquist_trunc = get_nyquist_mask(fs_trunc)
    
    overlap_loss_trunc = compute_overlap_loss(
        hp_rip_trunc * nyquist_trunc, hp_lal_trunc * nyquist_trunc,
        psd_trunc, fs_trunc
    )
    print(f"Overlap loss truncated at {f_cutoff:.1f} Hz: {float(overlap_loss_trunc):.15e}")
    print(f"log10(overlap_loss): {np.log10(float(overlap_loss_trunc)):.4f}")
    print()
    
    # Check overlap loss at various cutoffs
    print(f"=== Overlap loss at various frequency cutoffs ===")
    for f_cut in [500, 1000, 1500, 2000, 2200, 2400, 2500, 2600]:
        mask = fs_np < f_cut
        if mask.sum() < 10:
            continue
        fs_c = fs[mask]
        hp_lal_c = jnp.array(hp_lal)[mask]
        hp_rip_c = hp_ripple[mask]
        psd_c = psd_interp[mask]
        nyq_c = get_nyquist_mask(fs_c)
        
        ol = compute_overlap_loss(hp_rip_c * nyq_c, hp_lal_c * nyq_c, psd_c, fs_c)
        print(f"  f_cut = {f_cut:.0f} Hz: overlap_loss = {float(ol):.15e}, log10 = {np.log10(float(ol)):.4f}")


if __name__ == "__main__":
    main()
