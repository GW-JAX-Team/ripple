#!/usr/bin/env python
"""Check if LAL uses a different merger frequency for the Planck taper."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, MPC, PI, TWO_PI, MRSUN
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from tests.utils import get_freqs, get_lal_waveform

jax.config.update("jax_enable_x64", True)


def compute_merger_freq_ripple(m1, m2, kappa):
    """Ripple's merger frequency."""
    q = m1 / m2
    a_0 = 0.3586
    n_1 = 3.35411203e-2
    n_2 = 4.31460284e-5
    d_1 = 7.54224145e-2
    d_2 = 2.23626859e-4
    kappa_2 = kappa * kappa
    num = 1.0 + n_1 * kappa + n_2 * kappa_2
    den = 1.0 + d_1 * kappa + d_2 * kappa_2
    Q_0 = a_0 / np.sqrt(q)
    Momega_merger = Q_0 * (num / den)
    fHz_merger = Momega_merger / (TWO_PI) / ((m1 + m2) * MTSUN)
    return fHz_merger


def main():
    m1, m2 = 1.4, 1.3
    chi1, chi2 = 0.02, -0.01
    lambda1, lambda2 = 400.0, 300.0
    
    # LAL kappa computation (same as Ripple)
    M = m1 + m2
    X1 = m1 / M
    X2 = m2 / M
    term1 = (1.0 + 12.0 * X2 / X1) * (X1**5.0) * lambda1
    term2 = (1.0 + 12.0 * X1 / X2) * (X2**5.0) * lambda2
    kappa = (3.0 / 13.0) * (term1 + term2)
    
    f_merger_ripple = compute_merger_freq_ripple(m1, m2, kappa)
    print(f"Ripple f_merger: {f_merger_ripple:.1f} Hz")
    print(f"1.2 * f_merger:  {1.2 * f_merger_ripple:.1f} Hz")
    print()
    
    # Now let's check what LAL actually does
    # In LALSimIMRPhenomD_NRTidal.c, IMRPhenomD_NRTidal_Core:
    # const double fHz_mrg = XLALSimNRTunedTidesMergerFrequency( (m1_SI+m2_SI)/LAL_MSUN_SI, kappa2T, m1_SI/m2_SI);
    # const double NRTIDAL_FMAX = 1.3*fHz_mrg;
    # 
    # But wait, for IMRPhenomD_NRTidalv2, the waveform is generated using 
    # XLALSimIMRPhenomDFrequencySequence which generates IMRPhenomD up to the user's f_u.
    # Then the NRTidal corrections are applied.
    #
    # The Planck taper uses fHz_mrg from XLALSimNRTunedTidesMergerFrequency.
    # But does LAL use the SAME f_merger for both the taper and the fcut?
    #
    # Looking at the code flow:
    # 1. IMRPhenomD_NRTidal_Core is called
    # 2. It calls XLALSimIMRPhenomDFrequencySequence to get the BBH waveform
    # 3. Then it computes fHz_mrg = XLALSimNRTunedTidesMergerFrequency
    # 4. Then it calls XLALSimNRTunedTidesFDTidalPhaseFrequencySeries
    #    which computes phi_tidal, amp_tidal, planck_taper
    # 5. Then it assembles: data[j] *= planck_taper * exp(-i*phi_tidal - i*spin_phase)
    #
    # So LAL uses the same f_merger for the taper. Let me check if there's a difference
    # in how f_merger is computed.
    
    # Wait! Looking at the LAL code more carefully:
    # In IMRPhenomD_NRTidal_Core, when deltaF > 0 (uniform frequencies):
    #   double f_max_nr_tidal = fHigh;
    #   const double kappa2T = XLALSimNRTunedTidesComputeKappa2T(m1_SI, m2_SI, lambda1, lambda2);
    #   const double fHz_mrg = XLALSimNRTunedTidesMergerFrequency( (m1_SI+m2_SI)/LAL_MSUN_SI, kappa2T, m1_SI/m2_SI);
    #   const double NRTIDAL_FMAX = 1.3*fHz_mrg;
    #   
    #   if ( ( fHigh > NRTIDAL_FMAX ) || ( fHigh == 0.0 ) ) {
    #       f_max_nr_tidal = NRTIDAL_FMAX;
    #   }
    #   
    #   ret = XLALSimIMRPhenomDGenerateFD(..., fLow, f_max_nr_tidal, ...)
    #
    # This means LAL generates the BBH waveform only up to NRTIDAL_FMAX = 1.3*f_merger!
    # Then it resizes to fHigh with zeros.
    #
    # But when called via FrequencySequence (deltaF = 0), it uses:
    #   ret = XLALSimIMRPhenomDFrequencySequence(..., freqs_in, ...)
    # Which evaluates at ALL frequencies in freqs_in.
    #
    # So for FrequencySequence, the BBH waveform is NOT truncated at NRTIDAL_FMAX.
    # Only the taper is applied based on f_merger.
    
    # But wait, in the test we're using get_lal_waveform which calls
    # SimInspiralChooseFDWaveform with df, f_l, f_u. That's the uniform frequency case.
    # So LAL truncates the BBH waveform at 1.3*f_merger!
    
    # Let me compute what NRTIDAL_FMAX would be:
    NRTIDAL_FMAX = 1.3 * f_merger_ripple
    print(f"NRTIDAL_FMAX = 1.3 * f_merger = {NRTIDAL_FMAX:.1f} Hz")
    print()
    
    # This explains the amplitude decrease! LAL truncates the BBH waveform at NRTIDAL_FMAX,
    # so frequencies above that are zero. But the test frequencies only go up to 4096 Hz,
    # and NRTIDAL_FMAX = 2621 Hz. So frequencies between 2621 and 4096 should be zero in LAL.
    
    # But wait, looking at the code again:
    # if (fHigh > NRTIDAL_FMAX) {
    #     *htilde = XLALResizeCOMPLEX16FrequencySeries(*htilde, 0, n_full);
    # }
    # This resizes to fHigh but doesn't zero the data - it just extends with zeros.
    # So the data from fLow to NRTIDAL_FMAX is preserved, and from NRTIDAL_FMAX to fHigh is zero.
    
    # But our test uses f_u = 4096 Hz, and NRTIDAL_FMAX = 2621 Hz.
    # So frequencies above 2621 Hz should be zero in LAL's output.
    
    # Let me verify this
    theta_lal = np.array([m1, m2, chi1, chi2, lambda1, lambda2, 100.0, 0.0, 0.0, np.pi/4])
    
    f_l, f_u, f_sampling, T, f_ref = 20.0, 4096.0, 8192.0, 128.0, 20.0
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = float(fs[1] - fs[0])
    fs_np = np.array(fs)
    
    hp_lal, hc_lal = get_lal_waveform(
        theta_lal, "IMRPhenomD_NRTidalv2",
        f_l, f_u, df, f_ref,
        is_tidal=True, is_precessing=False
    )
    
    # Check amplitude at different frequencies
    test_freqs = np.array([500.0, 1000.0, 2000.0, 2500.0, 2621.0, 2700.0, 3000.0, 4000.0])
    print(f"=== LAL NRTidalv2 amplitude at select frequencies ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        amp = np.abs(hp_lal[idx])
        print(f"  f={tf:.0f} Hz: |hp| = {amp:.15e}")
    print()
    
    # If LAL zeros frequencies above NRTIDAL_FMAX, then amplitude at 2700 and 3000 should be ~0
    # Let me also check the BBH waveform
    
    hp_lal_bbh, hc_lal_bbh = get_lal_waveform(
        np.array([m1, m2, chi1, chi2, 100.0, 0.0, 0.0, np.pi/4]),
        "IMRPhenomD", f_l, f_u, df, f_ref,
        is_tidal=False, is_precessing=False
    )
    
    print(f"=== LAL BBH amplitude at select frequencies ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        amp = np.abs(hp_lal_bbh[idx])
        print(f"  f={tf:.0f} Hz: |hp| = {amp:.15e}")
    print()
    
    print(f"=== Key insight ===")
    print(f"f_merger = {f_merger_ripple:.1f} Hz")
    print(f"NRTIDAL_FMAX = 1.3 * f_merger = {NRTIDAL_FMAX:.1f} Hz")
    print()
    print("LAL truncates BBH waveform at NRTIDAL_FMAX when generating with uniform frequencies.")
    print("But when using FrequencySequence (which is what Ripple does internally),")
    print("LAL evaluates at all frequencies without truncation.")
    print()
    print("However, in our test, we're using SimInspiralChooseFDWaveform which uses")
    print("uniform frequencies, so LAL truncates at NRTIDAL_FMAX.")
    print()
    print("But wait - looking at the amplitude ratios earlier:")
    print("  At f=2000 Hz: ratio = 0.70")
    print("  f=2000 < NRTIDAL_FMAX=2621, so this is NOT due to truncation!")
    print()
    print("The amplitude decrease must be due to the Planck taper!")
    print("But our taper check showed taper = 1.0 at 2000 Hz...")
    print()
    print("Let me re-check: the taper in LAL is:")
    print("  planck_taper->data[i] = 1.0 - PlanckTaper(f, f_mrg, fHz_end_taper)")
    print("  where fHz_end_taper = 1.2 * f_mrg")
    print()
    print("Wait, for NRTidalv2NoAmpCorr_V, LAL uses:")
    print("  (*planck_taper).data[i] = 1.0 - PlanckTaper((*fHz).data[i], fHz_mrg, fHz_end_taper)")
    print()
    print("But f_mrg in LAL uses XLALSimNRTunedTidesMergerFrequency which uses LAL_MTSUN_SI")
    print("Let me check if there's a constant difference...")
    
    # Check LAL constants
    import lal
    print(f"\nLAL constants:")
    print(f"  LAL_MTSUN_SI = {lal.MTSUN_SI:.15e}")
    print(f"  LAL_MSUN_SI = {lal.MSUN_SI:.15e}")
    print()
    print(f"Ripple constants:")
    print(f"  MTSUN = {MTSUN:.15e}")
    print(f"  MRSUN = {MRSUN:.15e}")
    print()
    
    # Compute f_merger with LAL constants
    f_merger_lal = (
        0.3586 / np.sqrt(m1/m2)
        * (1.0 + 3.35411203e-2 * kappa + 4.31460284e-5 * kappa**2)
        / (1.0 + 7.54224145e-2 * kappa + 2.23626859e-4 * kappa**2)
        / (TWO_PI)
        / ((m1 + m2) * lal.MTSUN_SI)
    )
    print(f"f_merger with LAL constants: {f_merger_lal:.1f} Hz")
    print(f"f_merger with Ripple constants: {f_merger_ripple:.1f} Hz")
    print(f"Difference: {abs(f_merger_lal - f_merger_ripple):.4f} Hz")
    
    # Check if taper is the issue
    # In LAL, for uniform frequencies, the BBH waveform is generated up to NRTIDAL_FMAX
    # Then the taper is applied on top
    # But for frequencies between f_merger and NRTIDAL_FMAX, the taper goes from 1 to 0
    
    print(f"\n=== Planck taper values ===")
    f_mrg = f_merger_lal
    f_end = 1.2 * f_mrg
    for tf in test_freqs:
        # LAL's PlanckTaper: returns 0 for t <= t1, 1 for t >= t2
        # Then 1 - PlanckTaper is used, so it's 1 for f <= f_mrg, 0 for f >= 1.2*f_mrg
        if tf <= f_mrg:
            taper = 1.0
        elif tf >= f_end:
            taper = 0.0
        else:
            taper_val = np.exp((f_end - f_mrg) / (tf - f_mrg) + (f_end - f_mrg) / (tf - f_end))
            taper = 1.0 - 1.0 / (taper_val + 1.0)
        print(f"  f={tf:.0f} Hz: taper = {taper:.15e}")


if __name__ == "__main__":
    main()
