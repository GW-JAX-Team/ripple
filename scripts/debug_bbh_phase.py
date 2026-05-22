#!/usr/bin/env python
"""Check BBH phase matching between LAL and Ripple."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.conversions import ms_to_Mc_eta
from tests.utils import get_freqs, get_lal_waveform
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc

jax.config.update("jax_enable_x64", True)


def main():
    m1, m2 = 1.4, 1.3
    chi1, chi2 = 0.02, -0.01
    dist_mpc = 100.0
    tc = 0.0
    phic = 0.0
    inclination = np.pi / 4
    
    f_l, f_u, f_sampling, T, f_ref = 20.0, 4096.0, 8192.0, 128.0, 20.0
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = float(fs[1] - fs[0])
    fs_np = np.array(fs)
    
    # Generate BBH waveforms
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    bbh_theta = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])
    
    hp_bbh, hc_bbh = gen_IMRPhenomD_hphc(fs, bbh_theta, f_ref)
    hp_bbh_np = np.array(hp_bbh)
    
    hp_lal_bbh, hc_lal_bbh = get_lal_waveform(
        np.array([m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination]),
        "IMRPhenomD", f_l, f_u, df, f_ref,
        is_tidal=False, is_precessing=False
    )
    
    # Compare phases
    print(f"=== BBH phase comparison ===")
    test_freqs = np.array([50.0, 100.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0])
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        phase_lal = np.angle(hp_lal_bbh[idx])
        phase_rip = np.angle(hp_bbh_np[idx])
        phase_diff = phase_lal - phase_rip
        
        # Unwrap
        amp_lal = np.abs(hp_lal_bbh[idx])
        amp_rip = np.abs(hp_bbh_np[idx])
        
        print(f"  f={tf:.0f} Hz:")
        print(f"    |hp| LAL = {amp_lal:.15e}, Ripple = {amp_rip:.15e}, ratio = {amp_rip/amp_lal:.15f}")
        print(f"    phase LAL = {phase_lal:.15e}, Ripple = {phase_rip:.15e}")
        print(f"    phase diff = {phase_diff:.15e}")
    print()
    
    # Check overlap of BBH
    from tests.utils import get_nyquist_mask, compute_overlap
    psd_path = Path(__file__).parent.parent / "tests" / "psds" / "ET_D_psd.txt"
    psd_freqs_np, psd_np = np.loadtxt(psd_path, unpack=True)
    psd_interp = jnp.interp(fs, jnp.array(psd_freqs_np), jnp.array(psd_np))
    nyquist_mask = get_nyquist_mask(fs)
    
    overlap_bbh = compute_overlap(
        jnp.array(hp_bbh_np * nyquist_mask),
        jnp.array(hp_lal_bbh * nyquist_mask),
        psd_interp, fs
    )
    print(f"BBH overlap: {float(overlap_bbh):.15e}")
    print(f"BBH overlap loss: {1.0 - float(overlap_bbh):.15e}")
    print()
    
    # The BBH overlap loss is ~5e-11, which is much better than NRTidalv2's ~9e-5
    # So the issue is in the tidal corrections, not the BBH baseline
    
    # Let me check if the phase difference in NRTidalv2 is consistent
    # by comparing the phase difference pattern
    
    from ripplegw.conversions import lambdas_to_lambda_tildes
    from tests.utils import get_jitted_waveform
    
    lambda1, lambda2 = 400.0, 300.0
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
        jnp.array([lambda1, lambda2, m1, m2])
    )
    theta_ripple = jnp.array([
        Mc, eta, chi1, chi2, lambda_tilde, delta_lambda_tilde,
        dist_mpc, tc, phic, inclination
    ])
    
    theta_lal = np.array([m1, m2, chi1, chi2, lambda1, lambda2, dist_mpc, tc, phic, inclination])
    hp_lal_nr, hc_lal_nr = get_lal_waveform(
        theta_lal, "IMRPhenomD_NRTidalv2",
        f_l, f_u, df, f_ref,
        is_tidal=True, is_precessing=False
    )
    
    waveform = get_jitted_waveform("IMRPhenomD_NRTidalv2", fs, f_ref)
    hp_rip_nr, hc_rip_nr = waveform(theta_ripple)
    hp_rip_nr_np = np.array(hp_rip_nr)
    
    # Compare phase differences
    print(f"=== Phase difference comparison ===")
    print(f"NRTidalv2 phase diff = phase_NR - phase_BBH")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        phase_nr_lal = np.angle(hp_lal_nr[idx])
        phase_bbh_lal = np.angle(hp_lal_bbh[idx])
        phase_nr_rip = np.angle(hp_rip_nr_np[idx])
        phase_bbh_rip = np.angle(hp_bbh_np[idx])
        
        diff_lal = phase_nr_lal - phase_bbh_lal
        diff_rip = phase_nr_rip - phase_bbh_rip
        
        print(f"  f={tf:.0f} Hz:")
        print(f"    LAL: phase_NR - phase_BBH = {diff_lal:.15e}")
        print(f"    Ripple: phase_NR - phase_BBH = {diff_rip:.15e}")
        print(f"    Difference: {diff_lal - diff_rip:.15e}")


if __name__ == "__main__":
    main()
