#!/usr/bin/env python
"""Check if LAL's BBH phase in NRTidalv2 path differs from standalone BBH."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
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
    
    # Generate LAL BBH waveform (standalone)
    hp_lal_bbh, _ = get_lal_waveform(
        np.array([m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination]),
        "IMRPhenomD", f_l, f_u, df, f_ref,
        is_tidal=False, is_precessing=False
    )
    
    # Generate LAL NRTidalv2 waveform
    hp_lal_nr, _ = get_lal_waveform(
        theta_lal, "IMRPhenomD_NRTidalv2",
        f_l, f_u, df, f_ref,
        is_tidal=True, is_precessing=False
    )
    
    # Generate Ripple waveforms
    waveform = get_jitted_waveform("IMRPhenomD_NRTidalv2", fs, f_ref)
    hp_ripple, _ = waveform(theta_ripple)
    hp_ripple_np = np.array(hp_ripple)
    
    # Generate Ripple BBH
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
    bbh_theta = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])
    hp_bbh, _ = gen_IMRPhenomD_hphc(fs, bbh_theta, f_ref)
    hp_bbh_np = np.array(hp_bbh)
    
    # Key insight: LAL's NRTidalv2 phase should be:
    # phase_BBH_in_NRTidal - (psi_T + psi_SS)
    # where phase_BBH_in_NRTidal is the BBH phase when generated through the NRTidalv2 path.
    
    # We assumed phase_BBH_in_NRTidal = phase_standalone_BBH, but maybe they differ!
    
    # Let me compute the implied BBH phase from LAL's NRTidalv2:
    # phase_BBH_in_NRTidal = phase_NRTidal + (psi_T + psi_SS)
    
    from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import (
        get_tidal_phase,
        get_spin_phase_correction,
        get_kappa,
    )
    
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    kappa = float(get_kappa(theta_intrinsic))
    M = m1 + m2
    M_s = M * MTSUN
    
    test_freqs = np.array([50.0, 100.0, 500.0, 1000.0])
    
    print(f"=== Implied BBH phase from LAL NRTidalv2 ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        
        # LAL phases
        phase_lal_nr = np.angle(hp_lal_nr[idx])
        phase_lal_bbh = np.angle(hp_lal_bbh[idx])
        
        # Tidal corrections
        x_val = jnp.array([(PI * M_s * tf) ** (2.0 / 3.0)])
        psi_T = float(get_tidal_phase(x_val, theta_intrinsic, kappa)[0])
        psi_SS = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
        
        # Implied BBH phase from LAL NRTidalv2
        # phase_NR = phase_BBH_in_NR - (psi_T + psi_SS)
        # => phase_BBH_in_NR = phase_NR + (psi_T + psi_SS)
        phase_bbh_implied = phase_lal_nr + (psi_T + psi_SS)
        phase_bbh_implied_mod = ((phase_bbh_implied - phase_lal_bbh + PI) % (2*PI)) - PI
        
        print(f"f = {tf} Hz:")
        print(f"  phase_LAL_NR:          {phase_lal_nr:.15e}")
        print(f"  psi_T + psi_SS:        {psi_T + psi_SS:.15e}")
        print(f"  Implied BBH phase:     {phase_bbh_implied:.15e}")
        print(f"  LAL standalone BBH:    {phase_lal_bbh:.15e}")
        print(f"  Implied - Standalone:  {phase_bbh_implied_mod:.15e}")
        print()
    
    # If implied BBH phase differs from standalone BBH, then LAL uses a different
    # BBH phase in the NRTidalv2 path!
    
    # Also check Ripple
    print(f"=== Implied BBH phase from Ripple NRTidalv2 ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        
        # Ripple phases
        phase_rip_nr = np.angle(hp_ripple_np[idx])
        phase_rip_bbh = np.angle(hp_bbh_np[idx])
        
        # Tidal corrections
        x_val = jnp.array([(PI * M_s * tf) ** (2.0 / 3.0)])
        psi_T = float(get_tidal_phase(x_val, theta_intrinsic, kappa)[0])
        psi_SS = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
        
        # Implied BBH phase from Ripple NRTidalv2
        phase_bbh_implied = phase_rip_nr + (psi_T + psi_SS)
        phase_bbh_implied_mod = ((phase_bbh_implied - phase_rip_bbh + PI) % (2*PI)) - PI
        
        print(f"f = {tf} Hz:")
        print(f"  phase_Ripple_NR:       {phase_rip_nr:.15e}")
        print(f"  psi_T + psi_SS:        {psi_T + psi_SS:.15e}")
        print(f"  Implied BBH phase:     {phase_bbh_implied:.15e}")
        print(f"  Ripple standalone BBH: {phase_rip_bbh:.15e}")
        print(f"  Implied - Standalone:  {phase_bbh_implied_mod:.15e}")
        print()
    
    # If Ripple's implied BBH matches standalone BBH but LAL's doesn't,
    # then LAL uses a different BBH phase in the NRTidalv2 path.
    # This would explain the discrepancy!


if __name__ == "__main__":
    main()
