#!/usr/bin/env python
"""Debug remaining discrepancies between LAL and Ripple."""

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
    get_planck_taper,
    _get_merger_frequency,
    get_kappa,
)
from ripplegw.waveforms.IMRPhenomD import (
    Phase as BBH_Phase,
    Amp as BBH_Amp,
    get_IIb_raw_phase,
)
from ripplegw.waveforms.IMRPhenomD_utils import get_coeffs, get_transition_frequencies
from tests.utils import (
    get_freqs,
    get_jitted_waveform,
    get_lal_waveform,
    get_nyquist_mask,
    compute_overlap,
    compute_overlap_loss,
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
    
    # LAL parameter format
    theta_lal = np.array([m1, m2, chi1, chi2, lambda1, lambda2, dist_mpc, tc, phic, inclination])
    
    # Convert to Ripple format
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
        jnp.array([lambda1, lambda2, m1, m2])
    )
    theta_ripple = jnp.array([
        Mc, eta, chi1, chi2, lambda_tilde, delta_lambda_tilde,
        dist_mpc, tc, phic, inclination
    ])
    
    # Frequency setup
    f_l, f_u, f_sampling, T, f_ref = 20.0, 4096.0, 8192.0, 128.0, 20.0
    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = float(fs[1] - fs[0])
    fs_np = np.array(fs)
    
    print(f"Test parameters:")
    print(f"  m1={m1}, m2={m2}, chi1={chi1}, chi2={chi2}")
    print(f"  lambda1={lambda1}, lambda2={lambda2}")
    print(f"  n_freqs={len(fs)}, df={df:.6f} Hz")
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
    
    # BBH baseline comparison
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc as gen_bbh
    bbh_theta = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])
    hp_bbh, hc_bbh = gen_bbh(fs, bbh_theta, f_ref)
    hp_bbh_np = np.array(hp_bbh)
    
    hp_lal_bbh, hc_lal_bbh = get_lal_waveform(
        np.array([m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination]),
        "IMRPhenomD", f_l, f_u, df, f_ref,
        is_tidal=False, is_precessing=False
    )
    
    # Compare BBH
    test_freqs = np.array([50.0, 500.0, 1000.0, 2000.0])
    test_idx = [np.argmin(np.abs(fs_np - tf)) for tf in test_freqs]
    
    print(f"=== BBH baseline comparison ===")
    for tf, idx in zip(test_freqs, test_idx):
        amp_lal = np.abs(hp_lal_bbh[idx])
        amp_bbh = np.abs(hp_bbh_np[idx])
        phase_lal = np.angle(hp_lal_bbh[idx])
        phase_bbh = np.angle(hp_bbh_np[idx])
        print(f"  f={tf} Hz:")
        print(f"    |hp| LAL:    {amp_lal:.15e}")
        print(f"    |hp| Ripple: {amp_bbh:.15e}")
        print(f"    ratio:       {amp_bbh/amp_lal:.15f}")
        print(f"    phase LAL:    {phase_lal:.15e}")
        print(f"    phase Ripple: {phase_bbh:.15e}")
        print(f"    diff:         {phase_bbh - phase_lal:.15e}")
    print()
    
    # Now compare NRTidalv2 vs BBH in Ripple to see the tidal corrections
    print(f"=== NRTidalv2 / BBH ratio in Ripple ===")
    for tf, idx in zip(test_freqs, test_idx):
        ratio_amp = np.abs(hp_ripple_np[idx]) / np.abs(hp_bbh_np[idx])
        ratio_phase = np.angle(hp_ripple_np[idx]) - np.angle(hp_bbh_np[idx])
        print(f"  f={tf} Hz:")
        print(f"    |h_NRTidal| / |h_BBH| = {ratio_amp:.15e}")
        print(f"    phase_NRTidal - phase_BBH = {ratio_phase:.15e}")
    print()
    
    # Now compare NRTidalv2 / BBH ratio in LAL
    print(f"=== NRTidalv2 / BBH ratio in LAL ===")
    for tf, idx in zip(test_freqs, test_idx):
        ratio_amp = np.abs(hp_lal[idx]) / np.abs(hp_lal_bbh[idx])
        ratio_phase = np.angle(hp_lal[idx]) - np.angle(hp_lal_bbh[idx])
        print(f"  f={tf} Hz:")
        print(f"    |h_NRTidal| / |h_BBH| = {ratio_amp:.15e}")
        print(f"    phase_NRTidal - phase_BBH = {ratio_phase:.15e}")
    print()
    
    # Compute expected tidal phase correction
    M = m1 + m2
    M_s = M * MTSUN
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    kappa = float(get_kappa(theta_intrinsic))
    
    print(f"=== Expected tidal phase (psi_T) ===")
    for tf in test_freqs:
        x_val = jnp.array([(PI * M_s * tf) ** (2.0 / 3.0)])
        psi_T = float(get_tidal_phase(x_val, theta_intrinsic, kappa)[0])
        psi_SS = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
        print(f"  f={tf} Hz:")
        print(f"    psi_T = {psi_T:.15e}")
        print(f"    psi_SS = {psi_SS:.15e}")
        print(f"    psi_T + psi_SS = {psi_T + psi_SS:.15e}")
    print()
    
    # Check Planck taper
    f_merger = float(_get_merger_frequency(theta_intrinsic))
    print(f"=== Planck taper ===")
    print(f"  f_merger = {f_merger:.1f} Hz")
    print(f"  1.2*f_merger = {1.2*f_merger:.1f} Hz")
    for tf in test_freqs:
        taper = float(get_planck_taper(jnp.array([tf]), f_merger)[0])
        print(f"  f={tf} Hz: taper = {taper:.15e}")
    print()
    
    # The key question: what is the expected phase difference?
    # In LAL: phase_NRTidal = phase_BBH + psi_T + pn_fac*(SS_3p5PN + SSS_3p5PN)*f^(2/3)
    # In Ripple: phase_NRTidal = phase_BBH + psi_T + psi_SS
    
    # Let me check if the BBH phase reference handling is the same
    print(f"=== BBH phase reference handling ===")
    
    # In Ripple's gen_IMRPhenomD_NRTidalv2:
    # - It computes t0 from get_IIb_raw_phase
    # - It shifts phase: Psi -= t0 * ((f * M_s) - Mf_ref) + Psi_ref
    # - Then adds ext_phase_contrib = 2*PI*f*tc - 2*phic
    
    # In LAL's IMRPhenomD, the phase reference is handled differently
    # Let me check if the BBH phases actually match
    
    bbh_theta_intrinsic = jnp.array([m1, m2, chi1, chi2])
    coeffs = get_coeffs(bbh_theta_intrinsic)
    transition_freqs = get_transition_frequencies(bbh_theta_intrinsic, coeffs[5], coeffs[6])
    
    # Check t0 computation
    _, _, _, f4, f_RD, f_damp = transition_freqs
    t0_grad = jax.grad(get_IIb_raw_phase)(f4 * M_s, bbh_theta_intrinsic, coeffs, f_RD, f_damp)
    print(f"  t0 = {float(t0_grad):.15e}")
    print(f"  t0 * M_s = {float(t0_grad * M_s):.15e}")
    print()
    
    # Let me check if the phase difference at low frequency is constant
    print(f"=== Phase difference (NRTidal - BBH) across all frequencies ===")
    phase_diff = np.angle(hp_lal) - np.angle(hp_lal_bbh)
    print(f"  At f=20 Hz (idx=0): {phase_diff[0]:.15e}")
    print(f"  At f=100 Hz: {phase_diff[np.argmin(np.abs(fs_np - 100))]:.15e}")
    print(f"  At f=500 Hz: {phase_diff[np.argmin(np.abs(fs_np - 500))]:.15e}")
    print(f"  At f=1000 Hz: {phase_diff[np.argmin(np.abs(fs_np - 1000))]:.15e}")
    print(f"  At f=2000 Hz: {phase_diff[np.argmin(np.abs(fs_np - 2000))]:.15e}")
    print()
    
    # Check amplitude ratio across frequencies
    amp_ratio = np.abs(hp_lal) / np.abs(hp_lal_bbh)
    print(f"=== Amplitude ratio (|h_NRTidal| / |h_BBH|) across all frequencies ===")
    print(f"  At f=20 Hz (idx=0): {amp_ratio[0]:.15e}")
    print(f"  At f=100 Hz: {amp_ratio[np.argmin(np.abs(fs_np - 100))]:.15e}")
    print(f"  At f=500 Hz: {amp_ratio[np.argmin(np.abs(fs_np - 500))]:.15e}")
    print(f"  At f=1000 Hz: {amp_ratio[np.argmin(np.abs(fs_np - 1000))]:.15e}")
    print(f"  At f=2000 Hz: {amp_ratio[np.argmin(np.abs(fs_np - 2000))]:.15e}")
    print()
    
    # If amplitude ratio is not 1.0, then LAL IS applying some amplitude correction!
    # Let me check what version of NRTidal LAL actually uses
    
    import lalsimulation as lalsim
    
    # Check if there's an amplitude correction being applied
    # In LALSimIMRPhenomD_NRTidal.c, the waveform assembly is:
    # data[j] *= Corr
    # where Corr = planck_taper * exp(-i*phi_tidal - i*pn_fac*(SS_3p5PN + SSS_3p5PN)*f^(2/3))
    # This should NOT change amplitude (|Corr| = planck_taper)
    # But planck_taper should be 1.0 for frequencies below f_merger
    
    # Wait! The amplitude ratio IS changing! This means something else is different.
    # Let me check if the BBH waveforms themselves are different
    
    print(f"=== Direct BBH comparison (Ripple vs LAL) ===")
    nyquist_mask = get_nyquist_mask(fs)
    psd_path = Path(__file__).parent.parent / "tests" / "psds" / "ET_D_psd.txt"
    psd_freqs_np, psd_np = np.loadtxt(psd_path, unpack=True)
    psd_interp = jnp.interp(fs, jnp.array(psd_freqs_np), jnp.array(psd_np))
    
    overlap_bbh = compute_overlap(
        jnp.array(hp_bbh_np * nyquist_mask),
        jnp.array(hp_lal_bbh * nyquist_mask),
        psd_interp, fs
    )
    print(f"  BBH overlap: {float(overlap_bbh):.15e}")
    print(f"  BBH overlap loss: {1.0 - float(overlap_bbh):.15e}")
    print()
    
    # Check if BBH amplitude matches at specific frequencies
    print(f"=== BBH amplitude comparison at select frequencies ===")
    for tf, idx in zip(test_freqs, test_idx):
        amp_lal_bbh = np.abs(hp_lal_bbh[idx])
        amp_bbh = np.abs(hp_bbh_np[idx])
        print(f"  f={tf} Hz:")
        print(f"    |hp| LAL:    {amp_lal_bbh:.15e}")
        print(f"    |hp| Ripple: {amp_bbh:.15e}")
        print(f"    ratio:       {amp_bbh/amp_lal_bbh:.15f}")


if __name__ == "__main__":
    main()
