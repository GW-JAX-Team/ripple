#!/usr/bin/env python
"""Investigate the phase reference handling difference between LAL and Ripple."""

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
from ripplegw.waveforms.IMRPhenomD import (
    Phase as BBH_Phase,
    get_IIb_raw_phase,
)
from ripplegw.waveforms.IMRPhenomD_utils import get_coeffs, get_transition_frequencies
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
    
    f_l, f_u, f_sampling, T, f_ref = 20.0, 4096.0, 8192.0, 128.0, 20.0
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = float(fs[1] - fs[0])
    fs_np = np.array(fs)
    
    # Generate LAL BBH waveform
    hp_lal_bbh, _ = get_lal_waveform(
        np.array([m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination]),
        "IMRPhenomD", f_l, f_u, df, f_ref,
        is_tidal=False, is_precessing=False
    )
    
    # Compute Ripple's BBH phase
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    bbh_theta_intrinsic = jnp.array([m1, m2, chi1, chi2])
    coeffs = get_coeffs(bbh_theta_intrinsic)
    M_s = (m1 + m2) * MTSUN
    
    transition_freqs = get_transition_frequencies(bbh_theta_intrinsic, coeffs[5], coeffs[6])
    _, _, _, f4, f_RD, f_damp = transition_freqs
    
    # Compute t0
    t0 = jax.grad(get_IIb_raw_phase)(f4 * M_s, bbh_theta_intrinsic, coeffs, f_RD, f_damp)
    print(f"t0 = {float(t0):.15e}")
    print(f"t0 * M_s = {float(t0 * M_s):.15e}")
    
    # Compute BBH phase at f_ref
    Psi_ref = float(BBH_Phase(jnp.array(f_ref), bbh_theta_intrinsic, coeffs, transition_freqs))
    print(f"Psi_ref (at {f_ref} Hz) = {Psi_ref:.15e}")
    print()
    
    # Now let's reconstruct the BBH phase and compare with LAL
    # In Ripple: Psi = Phase(f) - t0*(f*M_s - Mf_ref) - Psi_ref + ext_phase
    # where ext_phase = 2*pi*f*tc - 2*phic
    
    # For tc=0, phic=0: ext_phase = 0
    # So Psi = Phase(f) - t0*(f*M_s - Mf_ref) - Psi_ref
    
    # Let me compute this at select frequencies
    test_freqs = np.array([50.0, 100.0, 500.0, 1000.0, 2000.0])
    
    print(f"=== Phase reconstruction comparison ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        phase_lal = np.angle(hp_lal_bbh[idx])
        
        # Ripple's phase
        Psi_f = float(BBH_Phase(jnp.array(tf), bbh_theta_intrinsic, coeffs, transition_freqs))
        Mf_ref = f_ref * M_s
        Psi = Psi_f - float(t0) * (tf * M_s - Mf_ref) - Psi_ref
        phase_ripple = -Psi  # h ~ exp(-i*Psi)
        
        # Phase difference (mod 2pi)
        phase_diff = phase_lal - phase_ripple
        phase_diff_mod = ((phase_diff + PI) % (2 * PI)) - PI
        
        print(f"  f={tf} Hz:")
        print(f"    phase LAL = {phase_lal:.15e}")
        print(f"    phase Ripple = {phase_ripple:.15e}")
        print(f"    diff = {phase_diff:.15e}")
        print(f"    diff mod 2pi = {phase_diff_mod:.15e}")
    print()
    
    # Now let's check the raw phase without t0 correction
    print(f"=== Raw phase comparison (without t0 correction) ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        phase_lal = np.angle(hp_lal_bbh[idx])
        
        Psi_f = float(BBH_Phase(jnp.array(tf), bbh_theta_intrinsic, coeffs, transition_freqs))
        phase_raw = -Psi_f
        
        phase_diff = phase_lal - phase_raw
        phase_diff_mod = ((phase_diff + PI) % (2 * PI)) - PI
        
        print(f"  f={tf} Hz:")
        print(f"    phase LAL = {phase_lal:.15e}")
        print(f"    phase raw = {phase_raw:.15e}")
        print(f"    diff = {phase_diff:.15e}")
        print(f"    diff mod 2pi = {phase_diff_mod:.15e}")
    print()
    
    # The key question: does LAL use the same t0 as Ripple?
    # LAL computes t0 from the IMRPhenomD phase at the merger-ringdown transition
    # Let me check if LAL's f4, f_RD, f_damp match Ripple's
    
    print(f"=== Transition frequencies ===")
    f4_ripple = float(f4)
    f_RD_ripple = float(f_RD)
    f_damp_ripple = float(f_damp)
    print(f"  f4 (Ripple) = {f4_ripple:.15e} Hz")
    print(f"  f_RD (Ripple) = {f_RD_ripple:.15e} Hz")
    print(f"  f_damp (Ripple) = {f_damp_ripple:.15e} Hz")
    print(f"  f4 * M_s (dimensionless) = {f4_ripple * M_s:.15e}")
    print()
    
    # Let me also check the phase reference frequency
    # In LAL, the phase is referenced to fRef, with phifRef = Phase(fRef)
    # Then the phase shift is: phi -= t0*(Mf - MfRef) + 2*phi0 + phifRef
    # where phi0 is the orbital phase at coalescence (passed as argument)
    
    # In Ripple, the phase shift is: Psi -= t0*(f*M_s - Mf_ref) + Psi_ref + ext_phase
    # where ext_phase = 2*pi*f*tc - 2*phic
    
    # For tc=0, phic=0: ext_phase = 0
    # And the waveform is h ~ exp(-i*Psi)
    
    # In LAL: h ~ amp0 * amp * exp(-i*phi)
    # where phi = IMRPhenDPhase - t0*(Mf - MfRef) + 2*phi0 + phifRef
    
    # The key difference might be in how phi0 is handled!
    # In LAL, phi0 is passed as an argument (phiRef in the function signature)
    # phiRef is the phase at reference time, which is 2*phic (orbital phase at coalescence)
    
    # Let me check what LAL uses for phi0
    # In get_lal_waveform, phi_ref = theta[6] = phic
    # And LAL calls SimInspiralChooseFDWaveform with phiRef
    
    # In LALSimIMRPhenomDFrequencySequence:
    # const REAL8 phi_precalc = 2.*phi0 + phifRef;
    # phi -= t0*(Mf-MfRef) + phi_precalc;
    
    # So the total phase shift is: t0*(Mf-MfRef) + 2*phi0 + phifRef
    # where phi0 = phiRef (passed argument) and phifRef = Phase(fRef)
    
    # In Ripple: Psi -= t0*(f*M_s - Mf_ref) + Psi_ref
    # So the phase shift is: t0*(f*M_s - Mf_ref) + Psi_ref
    
    # These should be the same if phi0 = phic = 0 and Psi_ref = phifRef
    
    print(f"=== Phase reference terms ===")
    print(f"LAL: phi_precalc = 2*phi0 + phifRef")
    print(f"  phi0 = phic = {phic}")
    print(f"  phifRef = Psi_ref = {Psi_ref:.15e}")
    print(f"  phi_precalc = {2*phic + Psi_ref:.15e}")
    print()
    print(f"Ripple: Psi_ref = {Psi_ref:.15e}")
    print(f"  ext_phase = 2*pi*f*tc - 2*phic = 0 (for tc=phic=0)")
    print()
    
    # So the phase reference handling should be identical for tc=phic=0.
    # Let me check if there's a difference in the raw phase computation.
    
    # Compute the raw phase at f_ref using LAL's internal function
    # This is tricky since we can't easily call LAL's internal IMRPhenDPhase
    # But we can check if the BBH phases match
    
    print(f"=== BBH phase difference (LAL vs Ripple) ===")
    print(f"This should be ~0 if everything matches")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        phase_lal = np.angle(hp_lal_bbh[idx])
        
        Psi_f = float(BBH_Phase(jnp.array(tf), bbh_theta_intrinsic, coeffs, transition_freqs))
        Mf_ref = f_ref * M_s
        Psi = Psi_f - float(t0) * (tf * M_s - Mf_ref) - Psi_ref
        phase_ripple = -Psi
        
        phase_diff = ((phase_lal - phase_ripple + PI) % (2 * PI)) - PI
        print(f"  f={tf} Hz: diff = {phase_diff:.15e}")
    print()
    
    # If the BBH phases match, then the issue in NRTidalv2 must be elsewhere.
    # Let me check if LAL's NRTidalv2 uses a different BBH phase than standalone IMRPhenomD.
    
    # Generate LAL NRTidalv2 waveform
    theta_lal = np.array([m1, m2, chi1, chi2, lambda1, lambda2, dist_mpc, tc, phic, inclination])
    hp_lal_nr, _ = get_lal_waveform(
        theta_lal, "IMRPhenomD_NRTidalv2",
        f_l, f_u, df, f_ref,
        is_tidal=True, is_precessing=False
    )
    
    print(f"=== NRTidalv2 vs BBH phase difference (LAL) ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        phase_nr = np.angle(hp_lal_nr[idx])
        phase_bbh = np.angle(hp_lal_bbh[idx])
        diff = ((phase_nr - phase_bbh + PI) % (2 * PI)) - PI
        print(f"  f={tf} Hz: diff = {diff:.15e}")
    print()
    
    # Compare with expected tidal phase
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    kappa = float(get_kappa(theta_intrinsic))
    
    print(f"=== Expected tidal phase (psi_T + psi_SS) ===")
    for tf in test_freqs:
        x_val = jnp.array([(PI * (m1 + m2) * MTSUN * tf) ** (2.0 / 3.0)])
        psi_T = float(get_tidal_phase(x_val, theta_intrinsic, kappa)[0])
        psi_SS = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
        print(f"  f={tf} Hz: psi_T = {psi_T:.15e}, psi_SS = {psi_SS:.15e}, sum = {psi_T + psi_SS:.15e}")
        print(f"  -(psi_T + psi_SS) = {-(psi_T + psi_SS):.15e}")
    print()
    
    # The key insight: LAL's NRTidalv2 phase difference (NR - BBH) should equal -(psi_T + psi_SS)
    # Let me check if this is the case
    
    print(f"=== Comparison: LAL phase diff vs -(psi_T + psi_SS) ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        phase_nr = np.angle(hp_lal_nr[idx])
        phase_bbh = np.angle(hp_lal_bbh[idx])
        diff_lal = ((phase_nr - phase_bbh + PI) % (2 * PI)) - PI
        
        x_val = jnp.array([(PI * (m1 + m2) * MTSUN * tf) ** (2.0 / 3.0)])
        psi_T = float(get_tidal_phase(x_val, theta_intrinsic, kappa)[0])
        psi_SS = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
        expected = -((psi_T + psi_SS + PI) % (2 * PI)) + PI  # modulo 2pi
        
        print(f"  f={tf} Hz:")
        print(f"    LAL phase diff = {diff_lal:.15e}")
        print(f"    -(psi_T+psi_SS) = {-(psi_T + psi_SS):.15e}")
        print(f"    diff = {diff_lal - (-(psi_T + psi_SS)):.15e}")


if __name__ == "__main__":
    main()
