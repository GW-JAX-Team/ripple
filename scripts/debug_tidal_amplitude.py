#!/usr/bin/env python
"""Deep debug of tidal amplitude discrepancy between LAL and Ripple."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, MPC, PI, TWO_PI, MRSUN

jax.config.update("jax_enable_x64", True)


def main():
    m1, m2 = 1.4, 1.3
    kappa = 66.95471594510244
    distance_mpc = 100.0
    f = 50.0
    
    M = m1 + m2
    M_s = M * MTSUN
    
    # Compute x
    x = (PI * M_s * f) ** (2.0 / 3.0)
    
    print(f"=== Deep tidal amplitude debug ===")
    print(f"m1={m1}, m2={m2}, M={M}")
    print(f"M_s = M * MTSUN = {M_s:.15e}")
    print(f"f = {f} Hz")
    print(f"x = (PI * M_s * f)^(2/3) = {x:.15e}")
    print()
    
    # LAL's computation of tidal amplitude (from SimNRTunedTidesFDTidalAmplitude)
    # This is the dimensionless amplitude
    prefac_lal = -9.0 * kappa
    poly_lal = (1.0 + 4.157407407407407 * x + 2519.111111111111 * x ** 2.89) / (1.0 + 13477.8073677 * x ** 4.0)
    ampT_lal_dim = prefac_lal * x ** (13.0 / 4.0) * poly_lal
    
    print(f"LAL dimensionless tidal amplitude:")
    print(f"  prefac = -9.0 * kappa = {prefac_lal:.15e}")
    print(f"  poly = {poly_lal:.15e}")
    print(f"  x^(13/4) = {x ** (13.0 / 4.0):.15e}")
    print(f"  ampT_dim = {ampT_lal_dim:.15e}")
    print()
    
    # Ripple's computation of amp0
    distance_m = distance_mpc * MPC
    amp0 = 2.0 * jnp.sqrt(5.0 / (64.0 * PI)) * M * MRSUN * M * MTSUN / distance_m
    print(f"Ripple amp0:")
    print(f"  distance_m = {distance_m:.15e}")
    print(f"  M * MRSUN = {M * MRSUN:.15e}")
    print(f"  M * MTSUN = {M * MTSUN:.15e}")
    print(f"  amp0 = {float(amp0):.15e}")
    print()
    
    # Full tidal amplitude in Ripple
    ampT_ripple = float(amp0 * 2 * jnp.sqrt(PI / 5) * prefac_lal * x ** (13.0 / 4.0) * poly_lal)
    print(f"Ripple full tidal amplitude:")
    print(f"  amp0 * 2 * sqrt(PI/5) = {float(amp0 * 2 * jnp.sqrt(PI / 5)):.15e}")
    print(f"  ampT = {ampT_ripple:.15e}")
    print()
    
    # Now let's check what LAL actually returns
    # In LAL, the tidal amplitude is NOT multiplied by anything - it's just the dimensionless value
    # But when applied to the waveform, it's combined differently
    
    # Let me check: what does the diagnostic script compute as LAL tidal amplitude?
    # compute_tidal_amplitude_lal returns: prefac * x^(13/4) * poly (dimensionless)
    
    print(f"=== Comparison ===")
    print(f"LAL dimensionless ampT: {ampT_lal_dim:.15e}")
    print(f"Ripple full ampT:       {ampT_ripple:.15e}")
    print()
    
    # Let me also compute what the BBH amplitude is at this frequency
    from ripplegw.waveforms.IMRPhenomD import Amp as BBH_Amp
    from ripplegw.waveforms.IMRPhenomD_utils import get_coeffs, get_transition_frequencies
    
    chi1, chi2 = 0.02, -0.01
    bbh_theta = jnp.array([m1, m2, chi1, chi2])
    coeffs = get_coeffs(bbh_theta)
    transition_freqs = get_transition_frequencies(bbh_theta, coeffs[5], coeffs[6])
    
    # Create a single frequency array
    f_arr = jnp.array([f])
    bbh_amp = BBH_Amp(f_arr, bbh_theta, coeffs, transition_freqs, D=distance_mpc)
    print(f"BBH amplitude at {f} Hz: {float(jnp.abs(bbh_amp[0])):.15e}")
    print()
    
    # The key question: what is the ratio A_T / bbh_amp?
    # In LAL: the tidal correction is multiplicative (planck * exp(-i*phi_tidal))
    # So the amplitude doesn't change from BBH (ignoring higher-order spin terms which only affect phase)
    # In Ripple: the tidal correction is additive (bbh_amp + A_T)
    # So the ratio is (bbh_amp + A_T) / bbh_amp = 1 + A_T / bbh_amp
    
    print(f"=== Key ratios ===")
    print(f"A_T / bbh_amp (Ripple) = {ampT_ripple / float(jnp.abs(bbh_amp[0])):.15e}")
    print(f"If this is ~0, then A_T is negligible in Ripple")
    print()
    
    # Now let's check what LAL's amplitude correction actually is
    # Looking at LALSimIMRPhenomD_NRTidal.c, for NRTidalv2_V:
    #   XLALSimNRTunedTidesFDTidalPhaseFrequencySeries(..., NRTidalv2NoAmpCorr_V)
    #   This computes tidal phase but NOT amplitude!
    
    # Wait - the LAL code uses NRTidalv2NoAmpCorr_V for IMRPhenomD_NRTidalv2!
    # Let me check what that means...
    
    print("=== LAL version check ===")
    print("In LALSimIMRPhenomD_NRTidal.c, IMRPhenomD_NRTidalv2 uses:")
    print("  XLALSimNRTunedTidesFDTidalPhaseFrequencySeries(..., NRTidalv2NoAmpCorr_V)")
    print("NRTidalv2NoAmpCorr_V means: tidal phase YES, tidal amplitude NO")
    print()
    print("In LALSimNRTunedTides.c, for NRTidalv2NoAmpCorr_V:")
    print("  for(UINT4 i = 0; i < (*fHz).length; i++) {")
    print("    (*phi_tidal).data[i] = SimNRTunedTidesFDTidalPhase_v2(...);")
    print("    (*planck_taper).data[i] = 1.0 - PlanckTaper(...);")
    print("  }")
    print("Notice: NO amp_tidal assignment!")
    print()
    print("This means LAL does NOT add tidal amplitude for IMRPhenomD_NRTidalv2!")
    print("LAL only applies: planck_taper * exp(-i * phi_tidal - i * spin_phase)")
    print()
    print("But Ripple DOES add tidal amplitude:")
    print("  h0 = A_P * (bbh_amp + A_T) * exp(...)")
    print()
    print("THIS IS THE BUG! Ripple adds A_T but LAL does not!")
    
    # Let me verify this by looking at what version LAL actually uses
    # In LALSimIMRPhenomD_NRTidal.c line ~213-219:
    # if (NRTidal_version == NRTidalv2_V) {
    #   ret = XLALSimNRTunedTidesFDTidalPhaseFrequencySeries(..., NRTidalv2NoAmpCorr_V);
    # }
    # else {
    #   XLALSimNRTunedTidesFDTidalPhaseFrequencySeries(..., NRTidal_version);
    # }
    # 
    # But wait, this is in the NRTidal_version == NRTidalv2_V branch
    # And then it calls XLALSimNRTunedTidesFDTidalPhaseFrequencySeries with NRTidalv2NoAmpCorr_V
    
    # Actually looking at the code more carefully, the NRTidal_version parameter passed to 
    # IMRPhenomD_NRTidal_Core is what determines the version.
    # When called from XLALSimIMRPhenomDNRTidalFrequencySequence, NRTidal_version is passed through.
    # When called from lalsim via SimInspiralChooseFDWaveform, NRTidal_version is determined by the approximant.
    
    # Let me check what version is actually used when we call lal with IMRPhenomD_NRTidalv2
    
    print("\n=== Verifying LAL version used ===")
    import lalsimulation as lalsim
    print(f"NRTidal_V = {lalsim.NRTidal_V}")
    print(f"NRTidalv2_V = {lalsim.NRTidalv2_V}")
    print(f"NRTidalv2NoAmpCorr_V = {lalsim.NRTidalv2NoAmpCorr_V}")
    
    # The key is: when we call get_lal_waveform with "IMRPhenomD_NRTidalv2",
    # LAL internally calls the NRTidalv2 version. But does it use NRTidalv2_V or NRTidalv2NoAmpCorr_V?
    
    # Looking at the LAL source code for how waveforms are called:
    # In LALSimInspiralChooseFDWaveform, the approximant is determined by the string.
    # For IMRPhenomD_NRTidalv2, it calls XLALSimIMRPhenomDNRTidalFrequencySequence with NRTidalv2_V.
    
    # But wait, looking at IMRPhenomD_NRTidal_Core:
    # if (NRTidal_version == NRTidalv2_V) {
    #   ret = XLALSimNRTunedTidesFDTidalPhaseFrequencySeries(phi_tidal, amp_tidal, planck_taper, freqs, ..., NRTidalv2NoAmpCorr_V);
    # }
    # This means even when NRTidalv2_V is requested, it calls the core with NRTidalv2NoAmpCorr_V!
    
    # Then after that, the waveform is assembled:
    # Corr = planck_taper->data[i] * cexp(-I*phi_tidal->data[i] - I*pn_fac*(SS_3p5PN + SSS_3p5PN)*pow(f,2./3.));
    # data[j] *= Corr;
    # Notice: NO amp_tidal is used in the Corr factor!
    
    # So LAL's IMRPhenomD_NRTidalv2 does NOT include tidal amplitude corrections.
    # It only includes tidal phase corrections + higher-order spin phase corrections + Planck taper.
    
    # But Ripple includes tidal amplitude. This is the bug!
    
    print("\n=== CONCLUSION ===")
    print("BUG FOUND: Ripple adds tidal amplitude (A_T) to the waveform,")
    print("but LAL's IMRPhenomD_NRTidalv2 does NOT include tidal amplitude.")
    print()
    print("LAL applies: planck_taper * exp(-i * phi_tidal - i * spin_phase)")
    print("Ripple applies: A_P * (bbh_amp + A_T) * exp(-i * (bbh_psi + psi_T + psi_SS))")
    print()
    print("To fix: Remove A_T from Ripple's waveform combination.")


if __name__ == "__main__":
    main()
