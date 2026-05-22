#!/usr/bin/env python
"""Final diagnostic: check if there's an issue with the phase sign convention or complex exponential."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import (
    get_tidal_phase,
    get_spin_phase_correction,
    get_kappa,
)
from tests.utils import get_freqs, get_lal_waveform, get_jitted_waveform

jax.config.update("jax_enable_x64", True)


def main():
    m1, m2 = 1.4, 1.3
    chi1, chi2 = 0.02, -0.01
    lambda1, lambda2 = 400.0, 300.0
    dist_mpc = 100.0
    tc = 0.0
    phic = 0.0
    inclination = np.pi / 4
    
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
        jnp.array([lambda1, lambda2, m1, m2])
    )
    theta_ripple = jnp.array([
        Mc, eta, chi1, chi2, lambda_tilde, delta_lambda_tilde,
        dist_mpc, tc, phic, inclination
    ])
    theta_lal = np.array([m1, m2, chi1, chi2, lambda1, lambda2, dist_mpc, tc, phic, inclination])
    
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
    hp_lal_bbh, hc_lal_bbh = get_lal_waveform(
        np.array([m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination]),
        "IMRPhenomD", f_l, f_u, df, f_ref,
        is_tidal=False, is_precessing=False
    )
    
    waveform = get_jitted_waveform("IMRPhenomD_NRTidalv2", fs, f_ref)
    hp_ripple, hc_ripple = waveform(theta_ripple)
    hp_ripple_np = np.array(hp_ripple)
    
    # Get BBH from Ripple
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
    bbh_theta = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])
    hp_bbh, hc_bbh = gen_IMRPhenomD_hphc(fs, bbh_theta, f_ref)
    hp_bbh_np = np.array(hp_bbh)
    
    # Compute expected tidal corrections
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    kappa = float(get_kappa(theta_intrinsic))
    M = m1 + m2
    M_s = M * MTSUN
    
    print(f"=== Phase decomposition ===")
    test_freqs = np.array([50.0, 100.0, 500.0, 1000.0])
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        
        # Phases from waveforms
        phase_lal_nr = np.angle(hp_lal[idx])
        phase_lal_bbh = np.angle(hp_lal_bbh[idx])
        phase_rip_nr = np.angle(hp_ripple_np[idx])
        phase_rip_bbh = np.angle(hp_bbh_np[idx])
        
        # Tidal corrections
        x_val = jnp.array([(PI * M_s * tf) ** (2.0 / 3.0)])
        psi_T = float(get_tidal_phase(x_val, theta_intrinsic, kappa)[0])
        psi_SS = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
        
        print(f"f = {tf} Hz:")
        print(f"  LAL NRTidal phase: {phase_lal_nr:.15e}")
        print(f"  LAL BBH phase:     {phase_lal_bbh:.15e}")
        print(f"  LAL diff (NR-BBH): {((phase_lal_nr - phase_lal_bbh + PI) % (2*PI)) - PI:.15e}")
        print(f"  Ripple NRTidal phase: {phase_rip_nr:.15e}")
        print(f"  Ripple BBH phase:     {phase_rip_bbh:.15e}")
        print(f"  Ripple diff (NR-BBH): {((phase_rip_nr - phase_rip_bbh + PI) % (2*PI)) - PI:.15e}")
        print(f"  -(psi_T + psi_SS):    {-(psi_T + psi_SS):.15e}")
        print()
    
    # Key insight: the phase diff (NR - BBH) should equal -(psi_T + psi_SS) mod 2pi
    # Let me check if LAL and Ripple both satisfy this
    
    print(f"=== Verification: phase_diff = -(psi_T + psi_SS) mod 2pi ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        
        x_val = jnp.array([(PI * M_s * tf) ** (2.0 / 3.0)])
        psi_T = float(get_tidal_phase(x_val, theta_intrinsic, kappa)[0])
        psi_SS = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
        expected = -(psi_T + psi_SS)
        
        phase_lal_nr = np.angle(hp_lal[idx])
        phase_lal_bbh = np.angle(hp_lal_bbh[idx])
        phase_rip_nr = np.angle(hp_ripple_np[idx])
        phase_rip_bbh = np.angle(hp_bbh_np[idx])
        
        diff_lal = ((phase_lal_nr - phase_lal_bbh + PI) % (2*PI)) - PI
        diff_rip = ((phase_rip_nr - phase_rip_bbh + PI) % (2*PI)) - PI
        expected_mod = ((expected + PI) % (2*PI)) - PI
        
        print(f"f = {tf} Hz:")
        print(f"  LAL diff (mod 2pi):    {diff_lal:.15e}")
        print(f"  Ripple diff (mod 2pi): {diff_rip:.15e}")
        print(f"  Expected (mod 2pi):    {expected_mod:.15e}")
        print(f"  LAL - Expected:        {((diff_lal - expected_mod + PI) % (2*PI)) - PI:.15e}")
        print(f"  Ripple - Expected:     {((diff_rip - expected_mod + PI) % (2*PI)) - PI:.15e}")
        print()


if __name__ == "__main__":
    main()
