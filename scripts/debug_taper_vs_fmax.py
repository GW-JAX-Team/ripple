#!/usr/bin/env python
"""Check if LAL uses NRTIDAL_FMAX for the taper instead of f_merger."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, MPC, PI, TWO_PI, MRSUN
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from tests.utils import get_freqs, get_lal_waveform, get_jitted_waveform
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import (
    _get_merger_frequency,
    get_planck_taper,
    get_tidal_phase,
    get_spin_phase_correction,
    get_kappa,
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
    
    # Compute amplitude and phase ratio
    amp_ratio = np.abs(hp_lal) / np.abs(hp_lal_bbh)
    phase_diff = np.angle(hp_lal) - np.angle(hp_lal_bbh)
    
    # Merger frequency
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    f_merger = float(_get_merger_frequency(theta_intrinsic))
    NRTIDAL_FMAX = 1.3 * f_merger
    f_end_taper = 1.2 * f_merger
    
    print(f"f_merger = {f_merger:.1f} Hz")
    print(f"NRTIDAL_FMAX = 1.3 * f_merger = {NRTIDAL_FMAX:.1f} Hz")
    print(f"f_end_taper = 1.2 * f_merger = {f_end_taper:.1f} Hz")
    print()
    
    # Check amplitude ratio at many frequencies
    print(f"=== Amplitude ratio (|h_NRTidal| / |h_BBH|) in LAL ===")
    test_freqs = np.array([100, 500, 1000, 1500, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600])
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        print(f"  f={tf:.0f} Hz: ratio = {amp_ratio[idx]:.10f}")
    print()
    
    # The key question: does LAL use f_merger or NRTIDAL_FMAX for the taper?
    # Looking at the LAL code again:
    # In IMRPhenomD_NRTidal_Core, after generating the BBH waveform:
    #   const double fHz_mrg = XLALSimNRTunedTidesMergerFrequency(...);
    #   const double fHz_end_taper = 1.2*fHz_mrg;
    #   ...
    #   (*planck_taper).data[i] = 1.0 - PlanckTaper((*fHz).data[i], fHz_mrg, fHz_end_taper);
    #
    # So LAL uses f_merger for the taper start, not NRTIDAL_FMAX.
    # The taper goes from 1 at f_merger to 0 at 1.2*f_merger.
    #
    # f_merger = 2016.7 Hz
    # 1.2*f_merger = 2420.0 Hz
    #
    # At f=2000 Hz (< f_merger), taper should be 1.0
    # But the amplitude ratio is 0.70!
    #
    # Wait... let me check if LAL actually generates IMRPhenomD up to NRTIDAL_FMAX
    # and then zeros everything beyond that. The ratio at f < f_merger should still be 1.0
    # because both NRTidalv2 and BBH have the same amplitude there.
    #
    # But the ratio is NOT 1.0! At f=2000 Hz, ratio = 0.70.
    # This can't be due to taper (taper=1.0 at 2000 Hz).
    # This can't be due to truncation (2000 < NRTIDAL_FMAX).
    #
    # Unless... the BBH waveform itself is generated differently!
    #
    # Wait - I need to check: when LAL generates IMRPhenomD_NRTidalv2, it generates
    # the BBH waveform up to NRTIDAL_FMAX. But when I call get_lal_waveform with
    # IMRPhenomD, it generates up to f_u = 4096 Hz!
    #
    # So the BBH waveforms are generated with different upper frequency limits!
    # This might cause subtle differences in the phase reference handling.
    
    print(f"=== Hypothesis check ===")
    print(f"When generating IMRPhenomD_NRTidalv2, LAL generates BBH up to NRTIDAL_FMAX = {NRTIDAL_FMAX:.1f} Hz")
    print(f"When generating IMRPhenomD separately, LAL generates up to f_u = {f_u} Hz")
    print(f"This difference in fHigh might affect the phase reference!")
    print()
    
    # Let me check if the issue is in how the BBH waveform is generated
    # by generating IMRPhenomD up to NRTIDAL_FMAX
    
    hp_lal_bbh_truncated, hc_lal_bbh_truncated = get_lal_waveform(
        np.array([m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination]),
        "IMRPhenomD", f_l, NRTIDAL_FMAX, df, f_ref,
        is_tidal=False, is_precessing=False
    )
    
    print(f"=== Comparing BBH generated up to f_u vs NRTIDAL_FMAX ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        if idx < len(hp_lal_bbh_truncated):
            amp_full = np.abs(hp_lal_bbh[idx])
            amp_trunc = np.abs(hp_lal_bbh_truncated[idx])
            print(f"  f={tf:.0f} Hz: |hp_full| = {amp_full:.15e}, |hp_trunc| = {amp_trunc:.15e}, ratio = {amp_trunc/amp_full:.15f}")
        else:
            print(f"  f={tf:.0f} Hz: beyond truncated array")
    print()
    
    # Now let me check the ratio of NRTidalv2 to BBH_truncated
    print(f"=== Amplitude ratio (|h_NRTidal| / |h_BBH_truncated|) in LAL ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        if idx < len(hp_lal_bbh_truncated):
            ratio = np.abs(hp_lal[idx]) / np.abs(hp_lal_bbh_truncated[idx])
            print(f"  f={tf:.0f} Hz: ratio = {ratio:.10f}")
    print()
    
    # If the ratio is now 1.0, then the issue was the BBH generation fHigh
    # Let me also check the phase
    print(f"=== Phase difference (NRTidal - BBH_truncated) in LAL ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        if idx < len(hp_lal_bbh_truncated):
            phase_diff_new = np.angle(hp_lal[idx]) - np.angle(hp_lal_bbh_truncated[idx])
            print(f"  f={tf:.0f} Hz: phase_diff = {phase_diff_new:.15e}")
    print()
    
    # Compare with expected tidal phase
    kappa = float(get_kappa(theta_intrinsic))
    print(f"=== Expected tidal phase corrections ===")
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        if idx < len(hp_lal_bbh_truncated):
            x_val = jnp.array([(PI * (m1 + m2) * MTSUN * tf) ** (2.0 / 3.0)])
            psi_T = float(get_tidal_phase(x_val, theta_intrinsic, kappa)[0])
            psi_SS = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
            print(f"  f={tf:.0f} Hz: psi_T = {psi_T:.6f}, psi_SS = {psi_SS:.6f}, sum = {psi_T + psi_SS:.6f}")
    print()
    
    # Now let's check if the issue is in how Ripple handles the BBH part
    print(f"=== Ripple NRTidalv2 vs LAL NRTidalv2 ===")
    waveform = get_jitted_waveform("IMRPhenomD_NRTidalv2", fs, f_ref)
    hp_ripple, hc_ripple = waveform(theta_ripple)
    hp_ripple_np = np.array(hp_ripple)
    
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        amp_lal_val = np.abs(hp_lal[idx])
        amp_rip_val = np.abs(hp_ripple_np[idx])
        phase_lal_val = np.angle(hp_lal[idx])
        phase_rip_val = np.angle(hp_ripple_np[idx])
        print(f"  f={tf:.0f} Hz:")
        print(f"    |hp| LAL = {amp_lal_val:.15e}, Ripple = {amp_rip_val:.15e}")
        print(f"    phase LAL = {phase_lal_val:.15e}, Ripple = {phase_rip_val:.15e}")
        print(f"    amp ratio = {amp_rip_val/amp_lal_val:.15f}")
        print(f"    phase diff = {phase_rip_val - phase_lal_val:.15e}")


if __name__ == "__main__":
    main()
