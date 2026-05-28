"""
Extract and compare the tidal phase psi_T between Ripple and LAL for q=0.25.

Strategy: Generate both waveforms with NO QM/SS corrections and chi=0,
then compare the phase arrays directly.

We can isolate the tidal phase by comparing:
  phi_total = phi_BBH + phi_tidal_correction
=> phi_tidal = phi_total - phi_BBH

We generate both the full NRTidalv3 waveform and the pure BBH waveform from
both LAL and Ripple, subtract to get the tidal phase contribution, then compare.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

try:
    import lal
    import lalsimulation as lalsim
except ImportError:
    sys.exit("LALSuite not available.")

from ripplegw.waveforms.NRTidalv3_utils import (
    get_tidalphasePN_coeffs,
    get_NRTidalv3_coefficients,
    get_tidal_phase,
    get_tidal_phase_PN,
    _get_merger_frequency,
    phenomx_tidal_phase,
    general_planck_taper,
    changePhase_if_min,
)
from ripplegw.constants import MTSUN, PI

# Test cases
M_TOT = 2.0
LAMBDA = 400.0
CHI = 0.0
DIST = 100.0
IOTA = 0.0

F_L = 20.0
F_U = 4096.0
T = 32.0
DF = 1.0 / T
F_REF = 20.0

fs = np.arange(F_L, F_U, DF)

def get_lal_phase(m1, m2, chi, l1, l2, approx_str):
    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    dist = DIST * 1e6 * lal.PC_SI
    laldict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, l1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, l2)
    q1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1)
    q2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, q1 - 1)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, q2 - 1)
    approx = lalsim.SimInspiralGetApproximantFromString(approx_str)
    hp, _ = lalsim.SimInspiralChooseFDWaveform(
        m1_kg, m2_kg, 0, 0, chi, 0, 0, chi,
        dist, IOTA, 0, 0, 0, 0,
        DF, F_L, F_U, F_REF, laldict, approx,
    )
    freq_arr = np.arange(len(hp.data.data)) * DF
    mask = (freq_arr >= F_L) & (freq_arr < F_U)
    hp_arr = hp.data.data[mask]
    n = min(len(hp_arr), len(fs))
    return hp_arr[:n]


for Q in [0.25, 1.00]:
    m1 = M_TOT / (1 + Q)
    m2 = M_TOT * Q / (1 + Q)
    l1, l2 = LAMBDA, LAMBDA

    print(f"\n===== q={Q:.2f}, m1={m1:.3f}, m2={m2:.3f} =====")

    # Get both waveforms from LAL
    hp_lal_tidal = get_lal_phase(m1, m2, CHI, l1, l2, "IMRPhenomXAS_NRTidalv3")
    hp_lal_bbh   = get_lal_phase(m1, m2, CHI, 0.0, 0.0, "IMRPhenomXAS")

    # Extract phases
    n = min(len(hp_lal_tidal), len(hp_lal_bbh), len(fs))
    hp_lal_tidal = hp_lal_tidal[:n]
    hp_lal_bbh   = hp_lal_bbh[:n]
    fc = fs[:n]

    # Amplitude threshold
    amp_thr = 0.01 * np.max(np.abs(hp_lal_tidal))
    valid = (np.abs(hp_lal_tidal) > amp_thr) & (np.abs(hp_lal_bbh) > amp_thr)

    phi_tidal_lal = np.unwrap(np.angle(hp_lal_tidal[valid]))
    phi_bbh_lal   = np.unwrap(np.angle(hp_lal_bbh[valid]))
    fv = fc[valid]

    # LAL tidal phase difference (tidal - bbh)
    delta_lal = phi_tidal_lal - phi_bbh_lal

    # Now compute Ripple's psi_T from the NRTidalv3_utils directly
    M_s = (m1 + m2) * MTSUN
    theta_intrinsic = jnp.array([m1, m2, CHI, CHI, l1, l2])
    Xa = m1 / (m1 + m2)

    PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
    NRv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)

    f_merger = float(_get_merger_frequency(theta_intrinsic))
    x_arr = jnp.array(PI * fv * M_s)

    # Ripple's psi_T (with changePhase_if_min and Planck taper)
    NRTidalv3_phase = np.array(get_tidal_phase(x_arr, NRv3_coeffs, PN_coeffs))
    PN_tidal_phase  = np.array(get_tidal_phase_PN(x_arr, Xa, float(l1), float(l2), PN_coeffs))

    fHzmrgcheck = 0.9 * f_merger
    increasing = np.concatenate([[False], NRTidalv3_phase[1:] >= NRTidalv3_phase[:-1]])
    valid_min = (fv >= fHzmrgcheck) & increasing
    if np.any(valid_min):
        idx_min = int(np.argmax(valid_min)) - 1
        idx_min = max(idx_min, 0)
        NRTidalv3_phase[idx_min:] = NRTidalv3_phase[idx_min]

    P_P = np.array(general_planck_taper(jnp.array(fv), 1.15*f_merger, 1.35*f_merger))
    psi_T = NRTidalv3_phase * (1 - P_P) + PN_tidal_phase * P_P

    # The phiTfRef contributes a reference offset. To compare apples-to-apples,
    # let's compute the difference between tidal phase at f and tidal phase at f_ref.
    Mf_ref = F_REF * M_s
    phiT_fref_ripple = float(phenomx_tidal_phase(theta_intrinsic, Mf_ref))
    # phenomx_tidal_phase at fref is psi_T_fref + QM+SS at fref. At chi=0, QM+SS=0.
    # So phiT_fref_ripple = psi_T at fref
    # For comparison with delta_lal, we need:
    # ripple_tidal_contribution(f) = phiT_fref - psi_T(f) + QM+SS at fref - QM+SS at f
    # At chi=0: = phiT_fref - psi_T(f) = psi_T_fref - psi_T(f)
    ripple_tidal_diff = phiT_fref_ripple - psi_T  # relative to fref

    # For LAL, the tidal contribution is also relative to fref:
    # phi_tidal_lal(f) - phi_tidal_lal(f_ref)
    # But delta_lal = phi_tidal_lal - phi_bbh_lal, which still has a linear drift (from tc shift)
    # Remove the linear+constant drift from delta_lal
    A = np.column_stack([fv, np.ones_like(fv)])
    coeffs, *_ = np.linalg.lstsq(A, delta_lal, rcond=None)
    delta_lal_detrended = delta_lal - A @ coeffs

    # Also detrend ripple
    coeffs_r, *_ = np.linalg.lstsq(A, ripple_tidal_diff, rcond=None)
    ripple_tidal_detrended = ripple_tidal_diff - A @ coeffs_r

    # Residual
    resid = ripple_tidal_detrended - delta_lal_detrended

    print(f"  RMS of detrended LAL tidal phase:    {np.sqrt(np.mean(delta_lal_detrended**2)):.4e} rad")
    print(f"  RMS of detrended Ripple tidal phase: {np.sqrt(np.mean(ripple_tidal_detrended**2)):.4e} rad")
    print(f"  RMS of residual (Ripple - LAL):      {np.sqrt(np.mean(resid**2)):.4e} rad")

    # Print raw phase values at a few test frequencies
    test_freqs = [50, 100, 200, 400]
    print(f"\n  Ripple NRTidalv3_coeffs:")
    print(f"    kappa2T-s10 = {float(NRv3_coeffs[0]) - 1.273000423:.6e}")
    print(f"    s1 = {float(NRv3_coeffs[0]):.6f}")
    print(f"    s2 = {float(NRv3_coeffs[1]):.6f}")
    print(f"    s3 = {float(NRv3_coeffs[2]):.6f}")
    print(f"    kappaA = {float(NRv3_coeffs[4]):.6f}")
    print(f"    kappaB = {float(NRv3_coeffs[5]):.6f}")

print("\nDone.")
