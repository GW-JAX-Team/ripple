#!/usr/bin/env python
"""Diagnostic script to compare intermediate values between LAL and Ripple for IMRPhenomD_NRTidalv2.

This script compares:
1. Kappa parameter
2. Merger frequency
3. Tidal phase
4. Tidal amplitude
5. Planck taper
6. Higher-order spin phase corrections
7. The final combined waveform

Usage:
    uv run scripts/debug_nrtidalv2_comparison.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.integrate import trapezoid

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, MPC, PI, TWO_PI, MRSUN
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.waveforms.IMRPhenom_tidal_utils import get_kappa, get_quadparam_octparam
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import (
    get_tidal_amplitude,
    get_tidal_phase,
    get_spin_phase_correction,
    get_planck_taper,
    _get_merger_frequency,
)
from tests.utils import (
    check_lal_available,
    get_freqs,
    get_jitted_waveform,
    get_lal_waveform,
    compute_overlap,
    compute_overlap_loss,
    generate_random_params,
    LAL_AVAILABLE,
)

jax.config.update("jax_enable_x64", True)


def compute_kappa_lal(m1, m2, lambda1, lambda2):
    """Compute kappa the same way LAL does it."""
    M = m1 + m2
    X1 = m1 / M
    X2 = m2 / M
    term1 = (1.0 + 12.0 * X2 / X1) * (X1**5.0) * lambda1
    term2 = (1.0 + 12.0 * X1 / X2) * (X2**5.0) * lambda2
    kappa = (3.0 / 13.0) * (term1 + term2)
    return kappa


def compute_merger_freq_lal(m1, m2, kappa):
    """Compute merger frequency the same way LAL does it."""
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


def compute_tidal_phase_lal(f, m1, m2, lambda1, lambda2, kappa):
    """Compute tidal phase the same way LAL does it (NRTidalv2)."""
    M = m1 + m2
    m_sec = M * MTSUN
    Xa = m1 / M
    Xb = m2 / M
    
    M_omega = PI * f * m_sec  # dimensionless angular GW frequency
    PN_x = M_omega ** (2.0 / 3.0)
    PN_x_2 = PN_x * PN_x
    PN_x_3 = PN_x * PN_x_2
    PN_x_3over2 = PN_x ** (3.0 / 2.0)
    PN_x_5over2 = PN_x ** (5.0 / 2.0)
    
    # NRTidalv2 coefficients
    c_Newt = 2.4375
    n_1 = -12.615214237993088
    n_3over2 = 19.0537346970349
    n_2 = -21.166863146081035
    n_5over2 = 90.55082156324926
    n_3 = -60.25357801943598
    d_1 = -15.111207827736678
    d_3over2 = 22.195327350624694
    d_2 = 8.064109635305156
    
    tidal_phase = -kappa * c_Newt / (Xa * Xb) * PN_x_5over2
    num = 1.0 + n_1 * PN_x + n_3over2 * PN_x_3over2 + n_2 * PN_x_2 + n_5over2 * PN_x_5over2 + n_3 * PN_x_3
    den = 1.0 + d_1 * PN_x + d_3over2 * PN_x_3over2 + d_2 * PN_x_2
    ratio = num / den
    tidal_phase *= ratio
    
    return tidal_phase


def compute_tidal_amplitude_lal(f, m1, m2, kappa, distance=1):
    """Compute tidal amplitude the same way LAL does it."""
    M = m1 + m2
    m_sec = M * MTSUN
    
    x = (PI * m_sec * f) ** (2.0 / 3.0)
    
    n1 = 4.157407407407407
    n289 = 2519.111111111111
    d = 13477.8073677
    poly = (1.0 + n1 * x + n289 * x ** 2.89) / (1.0 + d * x ** 4.0)
    
    prefac = -9.0 * kappa
    ampT = prefac * x ** (13.0 / 4.0) * poly
    
    return ampT


def compute_planck_taper_lal(f, f_merger):
    """Compute Planck taper the same way LAL does it."""
    f_end_taper = 1.2 * f_merger
    
    def planck_taper(t, t1, t2):
        if t <= t1:
            return 0.0
        elif t >= t2:
            return 1.0
        else:
            return 1.0 / (np.exp((t2 - t1) / (t - t1) + (t2 - t1) / (t - t2)) + 1.0)
    
    # LAL uses: 1.0 - PlanckTaper(f, f_merger, f_end_taper)
    # which is a taper that goes from 1 at low freq to 0 at high freq
    taper = np.array([1.0 - planck_taper(fi, f_merger, f_end_taper) for fi in f])
    return taper


def compute_spin_phase_lal(f, m1, m2, chi1, chi2, lambda1, lambda2):
    """Compute higher-order spin phase correction the same way LAL does it."""
    M = m1 + m2
    eta = m1 * m2 / M**2
    m_sec = M * MTSUN
    piM = PI * m_sec
    
    X_A = m1 / M
    X_B = m2 / M
    pn_fac = 3.0 * (piM) ** (2.0 / 3.0) / (128.0 * eta)
    
    # Get quadparams
    def get_quadparam_lal(lambda_):
        if lambda_ < 1:
            quadparam = 1.0 + lambda_ * (
                0.427688866723244
                + lambda_ * (-0.324336526985068 + lambda_ * 0.1107439432180572)
            )
            log_quadparam = np.log(quadparam)
            oct_coeffs = np.array([0.003131, 2.071, -0.7152, 0.2458, -0.03309])
            log_octparam = (
                oct_coeffs[0]
                + oct_coeffs[1] * log_quadparam
                + oct_coeffs[2] * log_quadparam**2
                + oct_coeffs[3] * log_quadparam**3
                + oct_coeffs[4] * log_quadparam**4
            )
            octparam = np.exp(log_octparam)
        else:
            quad_coeffs = np.array([0.1940, 0.09163, 0.04812, -4.283e-3, 1.245e-4])
            oct_coeffs = np.array([0.003131, 2.071, -0.7152, 0.2458, -0.03309])
            log_lambda = np.log(lambda_)
            log_quadparam = (
                quad_coeffs[0]
                + quad_coeffs[1] * log_lambda
                + quad_coeffs[2] * log_lambda**2
                + quad_coeffs[3] * log_lambda**3
                + quad_coeffs[4] * log_lambda**4
            )
            quadparam = np.exp(log_quadparam)
            log_octparam = (
                oct_coeffs[0]
                + oct_coeffs[1] * log_quadparam
                + oct_coeffs[2] * log_quadparam**2
                + oct_coeffs[3] * log_quadparam**3
                + oct_coeffs[4] * log_quadparam**4
            )
            octparam = np.exp(log_octparam)
        return quadparam, octparam
    
    quadparam1, octparam1 = get_quadparam_lal(lambda1)
    quadparam2, octparam2 = get_quadparam_lal(lambda2)
    
    # Higher-order spin terms (LAL XLALSimInspiralGetHOSpinTerms)
    chi1_sq = chi1**2
    chi2_sq = chi2**2
    X_Asq = X_A**2
    X_Bsq = X_B**2
    
    # Note: LAL passes quadparam+1 to the function, then subtracts 1 inside
    # So we need quadparam1 and quadparam2 as they are (not -1)
    SS_3p5PN = (-400.0 * PI * (quadparam1 - 1.0) * chi1_sq * X_Asq
                - 400.0 * PI * (quadparam2 - 1.0) * chi2_sq * X_Bsq)
    
    octparam1_minus_1 = (
        np.exp(
            0.003131 + 2.071 * np.log(quadparam1) - 0.7152 * np.log(quadparam1)**2
            + 0.2458 * np.log(quadparam1)**3 - 0.03309 * np.log(quadparam1)**4
        ) - 1.0
    )
    octparam2_minus_1 = (
        np.exp(
            0.003131 + 2.071 * np.log(quadparam2) - 0.7152 * np.log(quadparam2)**2
            + 0.2458 * np.log(quadparam2)**3 - 0.03309 * np.log(quadparam2)**4
        ) - 1.0
    )
    
    SSS_3p5PN = (
        10.0 * ((X_Asq + 308.0 / 3.0 * X_A) * chi1 + (X_Bsq - 89.0 / 3.0 * X_B) * chi2)
        * (quadparam1 - 1.0) * X_Asq * chi1_sq
        + 10.0 * ((X_Bsq + 308.0 / 3.0 * X_B) * chi2 + (X_Asq - 89.0 / 3.0 * X_A) * chi1)
        * (quadparam2 - 1.0) * X_Bsq * chi2_sq
        - 440.0 * octparam1_minus_1 * X_A * X_Asq * chi1_sq * chi1
        - 440.0 * octparam2_minus_1 * X_B * X_Bsq * chi2_sq * chi2
    )
    
    # The spin phase correction in LAL is: pn_fac * (SS_3p5PN + SSS_3p5PN) * f^(2/3)
    # where f^(2/3) comes from pow(f, 2./3.) in the loop
    # But note: in the loop, LAL uses pow(f, 2./3.) not pow(piM*f, 2./3.)
    
    spin_phase = pn_fac * (SS_3p5PN + SSS_3p5PN) * f ** (2.0 / 3.0)
    
    return spin_phase, SS_3p5PN, SSS_3p5PN, pn_fac


def main():
    if not LAL_AVAILABLE:
        print("ERROR: LALSuite is required for this diagnostic.")
        sys.exit(1)
    
    # Use a single test case
    m1, m2 = 1.4, 1.3  # Solar masses
    chi1, chi2 = 0.02, -0.01
    lambda1, lambda2 = 400.0, 300.0
    dist_mpc = 100.0
    tc = 0.0
    phic = 0.0
    inclination = np.pi / 4
    
    # LAL parameter format: [m1, m2, s1z, s2z, l1, l2, dist, tc, phic, inc]
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
    
    # === Compare intermediate quantities ===
    
    # 1. Kappa
    kappa_ripple = float(get_kappa(jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])))
    kappa_lal = compute_kappa_lal(m1, m2, lambda1, lambda2)
    print(f"1. Kappa:")
    print(f"   Ripple: {kappa_ripple:.15e}")
    print(f"   LAL:    {kappa_lal:.15e}")
    print(f"   Diff:   {abs(kappa_ripple - kappa_lal):.2e}")
    print()
    
    # 2. Merger frequency
    fmerger_ripple = float(_get_merger_frequency(jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])))
    fmerger_lal = compute_merger_freq_lal(m1, m2, kappa_lal)
    print(f"2. Merger frequency (Hz):")
    print(f"   Ripple: {fmerger_ripple:.15e}")
    print(f"   LAL:    {fmerger_lal:.15e}")
    print(f"   Diff:   {abs(fmerger_ripple - fmerger_lal):.2e}")
    print()
    
    # 3. Tidal phase (at a few frequencies)
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    kappa = kappa_ripple
    
    test_freqs = np.array([50.0, 500.0, 1000.0, 2000.0])
    print(f"3. Tidal phase at select frequencies:")
    for tf in test_freqs:
        psi_T_ripple = float(get_tidal_phase(
            jnp.array([(PI * (m1 + m2) * MTSUN * tf) ** (2.0 / 3.0)]),
            theta_intrinsic, kappa
        )[0])
        psi_T_lal = compute_tidal_phase_lal(tf, m1, m2, lambda1, lambda2, kappa_lal)
        print(f"   f={tf} Hz:")
        print(f"     Ripple: {psi_T_ripple:.15e}")
        print(f"     LAL:    {psi_T_lal:.15e}")
        print(f"     Diff:   {abs(psi_T_ripple - psi_T_lal):.2e}")
    print()
    
    # 4. Tidal amplitude
    print(f"4. Tidal amplitude (A_T) at select frequencies:")
    for tf in test_freqs:
        x_val = jnp.array([(PI * (m1 + m2) * MTSUN * tf) ** (2.0 / 3.0)])
        A_T_ripple = float(get_tidal_amplitude(x_val, theta_intrinsic, kappa, distance=dist_mpc)[0])
        A_T_lal = compute_tidal_amplitude_lal(tf, m1, m2, kappa_lal, distance=dist_mpc)
        print(f"   f={tf} Hz:")
        print(f"     Ripple: {A_T_ripple:.15e}")
        print(f"     LAL:    {A_T_lal:.15e}")
        print(f"     Diff:   {abs(A_T_ripple - A_T_lal):.2e}")
    print()
    
    # 5. Planck taper
    print(f"5. Planck taper at select frequencies:")
    for tf in test_freqs:
        taper_ripple = float(get_planck_taper(jnp.array([tf]), fmerger_ripple)[0])
        taper_lal = compute_planck_taper_lal(np.array([tf]), fmerger_lal)[0]
        print(f"   f={tf} Hz:")
        print(f"     Ripple: {taper_ripple:.15e}")
        print(f"     LAL:    {taper_lal:.15e}")
        print(f"     Diff:   {abs(taper_ripple - taper_lal):.2e}")
    print()
    
    # 6. Higher-order spin phase correction
    print(f"6. Higher-order spin phase correction at select frequencies:")
    for tf in test_freqs:
        x_val = jnp.array([(PI * (m1 + m2) * MTSUN * tf) ** (2.0 / 3.0)])
        psi_SS_ripple = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
        psi_SS_lal, SS_lal, SSS_lal, pn_fac_lal = compute_spin_phase_lal(tf, m1, m2, chi1, chi2, lambda1, lambda2)
        print(f"   f={tf} Hz:")
        print(f"     Ripple: {psi_SS_ripple:.15e}")
        print(f"     LAL:    {psi_SS_lal:.15e}")
        print(f"     Diff:   {abs(psi_SS_ripple - psi_SS_lal):.2e}")
    print()
    
    # 7. Full waveform comparison
    print(f"7. Full waveform comparison:")
    
    # Generate LAL waveform
    hp_lal, hc_lal = get_lal_waveform(
        theta_lal, "IMRPhenomD_NRTidalv2",
        f_l, f_u, df, f_ref,
        is_tidal=True, is_precessing=False
    )
    
    # Generate Ripple waveform
    waveform = get_jitted_waveform("IMRPhenomD_NRTidalv2", fs, f_ref)
    hp_ripple, hc_ripple = waveform(theta_ripple)
    hp_ripple_np = np.array(hp_ripple)
    hc_ripple_np = np.array(hc_ripple)
    
    # Compare amplitudes and phases at select frequencies
    test_idx = [np.argmin(np.abs(fs_np - tf)) for tf in test_freqs]
    print(f"   Amplitude |hp| at select frequencies:")
    for tf, idx in zip(test_freqs, test_idx):
        amp_lal = np.abs(hp_lal[idx])
        amp_ripple = np.abs(hp_ripple_np[idx])
        phase_lal = np.angle(hp_lal[idx])
        phase_ripple = np.angle(hp_ripple_np[idx])
        print(f"   f={tf} Hz (idx={idx}):")
        print(f"     |hp| LAL:    {amp_lal:.15e}")
        print(f"     |hp| Ripple: {amp_ripple:.15e}")
        print(f"     |hp| ratio:  {amp_ripple/amp_lal:.10f}")
        print(f"     phase LAL:    {phase_lal:.15e}")
        print(f"     phase Ripple: {phase_ripple:.15e}")
        print(f"     phase diff:   {phase_ripple - phase_lal:.15e}")
    
    # Compute overlap
    from tests.utils import get_nyquist_mask
    nyquist_mask = np.where(np.arange(len(fs_np)) < len(fs_np) - 2, 1.0, 0.0)
    
    psd_path = Path(__file__).parent.parent / "tests" / "psds" / "ET_D_psd.txt"
    psd_freqs_np, psd_np = np.loadtxt(psd_path, unpack=True)
    psd_interp = np.interp(fs_np, psd_freqs_np, psd_np)
    
    overlap = compute_overlap(
        jnp.array(hp_ripple_np * nyquist_mask),
        jnp.array(hp_lal * nyquist_mask),
        jnp.array(psd_interp),
        fs
    )
    overlap_loss = 1.0 - float(overlap)
    
    print(f"\n   Overlap: {float(overlap):.15e}")
    print(f"   Overlap loss: {overlap_loss:.15e}")
    print(f"   log10(overlap_loss): {np.log10(overlap_loss):.4f}")
    
    # Check if the BBH baseline agrees
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc as gen_bbh
    
    # Need to convert theta for BBH
    bbh_theta = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination])
    hp_bbh, hc_bbh = gen_bbh(fs, bbh_theta, f_ref)
    hp_bbh_np = np.array(hp_bbh)
    
    # Get LAL BBH waveform for comparison
    hp_lal_bbh, hc_lal_bbh = get_lal_waveform(
        np.array([m1, m2, chi1, chi2, dist_mpc, tc, phic, inclination]),
        "IMRPhenomD", f_l, f_u, df, f_ref,
        is_tidal=False, is_precessing=False
    )
    
    overlap_bbh = compute_overlap(
        jnp.array(hp_bbh_np * nyquist_mask),
        jnp.array(hp_lal_bbh * nyquist_mask),
        jnp.array(psd_interp),
        fs
    )
    print(f"\n   BBH baseline overlap: {float(overlap_bbh):.15e}")
    print(f"   BBH baseline overlap loss: {1.0 - float(overlap_bbh):.15e}")


if __name__ == "__main__":
    main()
