#!/usr/bin/env python
"""Check phase accumulation in LAL vs Ripple."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, MPC, PI, TWO_PI, MRSUN
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import (
    get_tidal_phase,
    get_spin_phase_correction,
    get_kappa,
)
from tests.utils import get_freqs, get_lal_waveform

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
    
    # Compare phases
    print(f"=== Phase comparison ===")
    test_freqs = np.array([50.0, 100.0, 500.0, 1000.0, 2000.0])
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        phase_lal = np.angle(hp_lal[idx])
        phase_bbh = np.angle(hp_lal_bbh[idx])
        phase_diff = phase_lal - phase_bbh
        
        # Expected tidal phase
        M = m1 + m2
        M_s = M * MTSUN
        theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
        kappa = float(get_kappa(theta_intrinsic))
        x_val = jnp.array([(PI * M_s * tf) ** (2.0 / 3.0)])
        psi_T = float(get_tidal_phase(x_val, theta_intrinsic, kappa)[0])
        psi_SS = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
        
        print(f"  f={tf} Hz:")
        print(f"    phase_LAL_NRTidal = {phase_lal:.15e}")
        print(f"    phase_LAL_BBH     = {phase_bbh:.15e}")
        print(f"    phase_diff        = {phase_diff:.15e}")
        print(f"    -psi_T            = {-psi_T:.15e}")
        print(f"    -psi_SS           = {-psi_SS:.15e}")
        print(f"    -(psi_T+psi_SS)   = {-(psi_T + psi_SS):.15e}")
        print(f"    phase_diff vs -(psi_T+psi_SS): {phase_diff - (-(psi_T + psi_SS)):.15e}")
    print()
    
    # The phase diff in LAL should be -(psi_T + psi_SS)
    # Let me check if there's a sign difference
    print(f"=== Sign check ===")
    print(f"In LAL: data[j] *= planck_taper * exp(-I*phi_tidal - I*pn_fac*(SS_3p5PN + SSS_3p5PN)*f^(2/3))")
    print(f"So the phase correction is: -(phi_tidal + pn_fac*(SS+SSS)*f^(2/3))")
    print(f"This should equal: -(psi_T + psi_SS)")
    print()
    
    # Check the phase accumulation more carefully
    # In LAL's IMRPhenomD_NRTidal_Core:
    # 1. First, IMRPhenomD is called which computes: amp * exp(-I * phi)
    #    where phi includes t0*(Mf-MfRef) + phi_precalc
    # 2. Then, the correction is applied: data[j] *= planck * exp(-I*phi_tidal - I*spin)
    #    So the final phase is: phi + phi_tidal + spin
    # 
    # But wait, the sign! In LAL: cexp(-I * phi) means phase = -phi
    # And cexp(-I*phi_tidal - I*spin) means additional phase = -(phi_tidal + spin)
    # So total phase = -(phi + phi_tidal + spin)
    # 
    # The phase difference between NRTidal and BBH should be:
    # -(phi_NR + phi_tidal + spin) - (-phi_BBH) = -(phi_NR - phi_BBH) - (phi_tidal + spin)
    # 
    # If phi_NR == phi_BBH (same BBH phase), then phase_diff = -(phi_tidal + spin)
    # But phi_tidal = psi_T from our computation, and spin = psi_SS
    # So phase_diff should be -(psi_T + psi_SS)
    
    # The diagnostic shows phase_diff is positive while -(psi_T+psi_SS) is also positive
    # But they don't match exactly. Let me check if there's a modulo 2*pi issue
    
    print(f"=== Modulo 2*pi check ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        phase_lal = np.angle(hp_lal[idx])
        phase_bbh = np.angle(hp_lal_bbh[idx])
        phase_diff = phase_lal - phase_bbh
        
        M = m1 + m2
        M_s = M * MTSUN
        theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
        kappa = float(get_kappa(theta_intrinsic))
        x_val = jnp.array([(PI * M_s * tf) ** (2.0 / 3.0)])
        psi_T = float(get_tidal_phase(x_val, theta_intrinsic, kappa)[0])
        psi_SS = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
        
        expected = -(psi_T + psi_SS)
        # Compute modulo 2*pi
        diff_mod = (phase_diff - expected) % (2 * PI)
        diff_mod_centered = ((phase_diff - expected + PI) % (2 * PI)) - PI
        
        print(f"  f={tf} Hz:")
        print(f"    phase_diff = {phase_diff:.15e}")
        print(f"    expected   = {expected:.15e}")
        print(f"    diff       = {phase_diff - expected:.15e}")
        print(f"    diff mod 2pi = {diff_mod:.15e}")
        print(f"    diff centered = {diff_mod_centered:.15e}")


if __name__ == "__main__":
    main()
