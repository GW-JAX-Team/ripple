"""Probe ripple's IMRPhenomXAS_NRTidalv3 phase decomposition against LAL internals.

Goal: localize the residual ~1e-8 overlap-loss to a specific phase component.

Compares, point-by-point on a Mf grid for a single representative BNS sample:
  (1) Merger frequency:    ripple _get_merger_frequency vs LAL SimNRTunedTidesMergerFrequency_v3
  (2) Tidal phase series:  ripple psi_T(f) vs LAL SimNRTunedTidesFDTidalPhaseFrequencySeries
  (3) Full BBH phase:      ripple "(bbh_psi + phase_shift)" vs LAL PhenomXOnlyReturnPhase

Run via:
  uv run --group cross-validation python scripts/probe_xas_nrtidalv3_phase.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import lal
import lalsimulation as lalsim

from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import ms_to_Mc_eta
from ripplegw.waveforms.NRTidalv3_utils import (
    _get_merger_frequency,
    fullTidalPhaseCorrection,
    general_planck_taper,
    get_NRTidalv3_coefficients,
    get_tidal_phase,
    get_tidal_phase_PN,
    get_tidalphasePN_coeffs,
    changePhase_if_min,
)
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import (
    get_qm_phase_correction,
    get_spin_phase_correction,
)
from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc


# ------------------------------------------------------------------------------
# Representative BNS sample (matches first cached LAL sample)
# ------------------------------------------------------------------------------
def get_sample():
    data = np.load("tests/cross_validation/lal_cache/IMRPhenomXAS_NRTidalv3_T128.npz",
                   allow_pickle=False)
    # use worst-mismatch sample (audit said idx=1)
    i = 1
    return data["theta_batch"][i]


def main():
    theta_lal = get_sample()
    # theta_lal = [m1, m2, s1z, s2z, l1, l2, dist, tc, phic, inc]
    m1, m2, s1z, s2z, l1, l2, dL, tc, phic, inc = theta_lal
    print(f"Sample: m1={m1:.3f} m2={m2:.3f} s1z={s1z:.4f} s2z={s2z:.4f}"
          f" l1={l1:.1f} l2={l2:.1f} dL={dL:.1f}")

    M = m1 + m2
    M_s = M * MTSUN
    theta_int = jnp.array([m1, m2, s1z, s2z, l1, l2])

    # --------------------------------------------------------------------------
    # (1) Merger frequency comparison
    # --------------------------------------------------------------------------
    f_mrg_ripple = float(_get_merger_frequency(theta_int))
    # LAL expects q>=1
    q = m1 / m2 if m1 >= m2 else m2 / m1
    f_mrg_lal = lalsim.SimNRTunedTidesMergerFrequency_v3(
        float(M), float(l1), float(l2), float(q), float(s1z), float(s2z)
    )
    print(f"\n(1) Merger freq:  ripple={f_mrg_ripple:.10f}  LAL={f_mrg_lal:.10f}"
          f"  rel-diff={(f_mrg_ripple - f_mrg_lal) / f_mrg_lal:.3e}")

    # --------------------------------------------------------------------------
    # (2) Per-frequency tidal phase comparison
    # --------------------------------------------------------------------------
    # Use a moderate-density grid spanning the band
    T = 128.0
    df = 1.0 / T
    fs = np.arange(20.0, 4096.0, df)

    # LAL: call XLALSimNRTunedTidesFDTidalPhaseFrequencySeries
    fHz_seq = lal.CreateREAL8Sequence(len(fs))
    fHz_seq.data[:] = fs
    phi_tidal_seq = lal.CreateREAL8Sequence(len(fs))
    amp_tidal_seq = lal.CreateREAL8Sequence(len(fs))
    planck_seq = lal.CreateREAL8Sequence(len(fs))
    # m1>=m2 enforced internally; pass SI masses
    m1_SI = float(m1) * lal.MSUN_SI
    m2_SI = float(m2) * lal.MSUN_SI
    lalsim.SimNRTunedTidesFDTidalPhaseFrequencySeries(
        phi_tidal_seq, amp_tidal_seq, planck_seq, fHz_seq,
        m1_SI, m2_SI, float(l1), float(l2), float(s1z), float(s2z),
        lalsim.NRTidalv3_V,
    )
    lal_phi_tidal = np.array(phi_tidal_seq.data)
    lal_planck_taper = np.array(planck_seq.data)
    print(f"\n(2a) LAL phi_tidal[idx] range: [{lal_phi_tidal.min():.4e},"
          f" {lal_phi_tidal.max():.4e}]")

    # Ripple equivalent: psi_T = NRTidalv3_phase * (1-P_P) + PNphase * P_P
    # where NRTidalv3_phase has min-clamp applied; P_P is general_planck_taper
    # between 1.15 and 1.35 * f_merger.
    Xa_jnp = m1 / M
    x_pi_f_Ms = PI * jnp.array(fs) * M_s
    PN_coeffs = get_tidalphasePN_coeffs(theta_int)
    NRT_coeffs = get_NRTidalv3_coefficients(theta_int, PN_coeffs)
    NR_phase = get_tidal_phase(x_pi_f_Ms, NRT_coeffs, PN_coeffs)
    fHzmrgcheck = 0.9 * f_mrg_ripple
    incr = jnp.concatenate([jnp.array([False]), NR_phase[1:] >= NR_phase[:-1]])
    valid = (jnp.array(fs) >= fHzmrgcheck) & incr
    NR_phase_clamped = jax.lax.cond(
        jnp.any(valid),
        lambda arr: changePhase_if_min(*arr),
        lambda arr: arr[1],
        (jnp.array(fs), NR_phase, valid),
    )
    P_P = general_planck_taper(jnp.array(fs), 1.15 * f_mrg_ripple, 1.35 * f_mrg_ripple)
    PN_phase = get_tidal_phase_PN(x_pi_f_Ms, Xa_jnp, l1, l2, PN_coeffs)
    ripple_phi_tidal_NRblend = NR_phase_clamped * (1 - P_P) + PN_phase * P_P

    # Note: LAL also adds spin-induced 2PN/3PN/3.5PN terms to phaseTidal in the
    # main loop. Ripple keeps these separate (psi_QM + psi_SS). For an
    # apples-to-apples we just compare the NR/PN-blend slice (which is what
    # LAL's SimNRTunedTidesFDTidalPhaseFrequencySeries returns).
    diff = np.array(ripple_phi_tidal_NRblend) - lal_phi_tidal
    print(f"(2b) ripple psi_T (NR/PN blend only) - LAL phi_tidal[idx]:")
    print(f"     max abs:    {np.abs(diff).max():.3e}")
    print(f"     RMS:        {np.sqrt(np.mean(diff**2)):.3e}")
    # band-resolved
    bands = [(20, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 4096)]
    for lo, hi in bands:
        m = (fs >= lo) & (fs < hi)
        if m.any():
            print(f"     {lo:>4}-{hi:<4} Hz: max|d|={np.abs(diff[m]).max():.3e}"
                  f"  RMS={np.sqrt(np.mean(diff[m]**2)):.3e}")

    # --------------------------------------------------------------------------
    # (3) Full ripple BBH-portion phase vs LAL PhenomXOnlyReturnPhase
    # --------------------------------------------------------------------------
    # Build LAL waveform with PhenomXOnlyReturnPhase=1 → htilde data is purely
    # phi(f) (BBH phase + linb*Mf + phifRef + lina), without tidal subtraction.
    f_l = 20.0
    f_u = 4096.0
    params_dict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXOnlyReturnPhase(params_dict, 1)
    # set tides
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params_dict, float(l1))
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params_dict, float(l2))
    # explicitly select NRTidalv3
    lalsim.SimInspiralWaveformParamsInsertPhenomXTidalFlag(params_dict, 3)
    f_ref = 20.0
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_SI, m2_SI, 0.0, 0.0, float(s1z), 0.0, 0.0, float(s2z),
        float(dL) * 1e6 * lal.PC_SI,
        0.0,  # inclination set to 0 for clean phase read
        0.0,  # phiRef
        0.0,  # longAscNodes
        0.0,  # eccentricity
        0.0,  # meanPerAno
        df, f_l, f_u, f_ref,
        params_dict,
        lalsim.IMRPhenomXAS_NRTidalv3,
    )
    # hp.data.data contains phi (real, but stored as complex); extract real part
    lal_full_phase_data = np.asarray(hp.data.data)
    # Frequency array for LAL
    lal_freqs = np.arange(hp.data.length) * hp.deltaF
    print(f"\n(3a) LAL phenom-only-return-phase length={len(lal_full_phase_data)}"
          f" freqs=[{lal_freqs[0]:.4f}, {lal_freqs[-1]:.4f}]")
    # Mask to test band [f_l, f_u]
    band_mask = (lal_freqs >= f_l) & (lal_freqs < f_u)
    fb = lal_freqs[band_mask]
    lal_phi_band = lal_full_phase_data[band_mask].real

    # Now compute ripple's "(bbh_psi + phase_shift) - 2pi*f*tc - 2*phic" without
    # the tidal subtraction, on the same grid fb.
    # The cleanest way: monkey-patch the no-taper path? Or just rebuild from
    # parts. For diagnostics, generate ripple's full hp and unwrap its arg.
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    from ripplegw.conversions import lambdas_to_lambda_tildes
    lam_t, dlam_t = lambdas_to_lambda_tildes(jnp.array([l1, l2, m1, m2]))
    rip_params = jnp.array([Mc, eta, s1z, s2z, lam_t, dlam_t, dL, 0.0, 0.0, 0.0])
    rip_hp, _ = gen_IMRPhenomXAS_NRTidalv3_hphc(
        jnp.array(fb), rip_params, f_ref
    )
    rip_phase_full = np.unwrap(np.angle(np.array(rip_hp)))

    # We can't isolate the BBH portion easily because ripple computes h0 with
    # tidal subtraction inline. But we CAN compare overall phase difference.
    # Note: LAL's PhenomXOnlyReturnPhase output is phi (no tidal sub), so the
    # comparable ripple quantity is rip_phase_full + (psi_T+psi_QM+psi_SS).
    Mf_band = jnp.array(fb) * M_s
    NR_phase_b = get_tidal_phase(PI * Mf_band, NRT_coeffs, PN_coeffs)
    incr_b = jnp.concatenate([jnp.array([False]), NR_phase_b[1:] >= NR_phase_b[:-1]])
    valid_b = (jnp.array(fb) >= fHzmrgcheck) & incr_b
    NR_phase_b_c = jax.lax.cond(
        jnp.any(valid_b),
        lambda arr: changePhase_if_min(*arr),
        lambda arr: arr[1],
        (jnp.array(fb), NR_phase_b, valid_b),
    )
    P_P_b = general_planck_taper(jnp.array(fb), 1.15 * f_mrg_ripple, 1.35 * f_mrg_ripple)
    PN_phase_b = get_tidal_phase_PN(PI * Mf_band, Xa_jnp, l1, l2, PN_coeffs)
    psi_T_b = NR_phase_b_c * (1 - P_P_b) + PN_phase_b * P_P_b
    f_Ms_b = jnp.array(fb) * M_s
    x_23_b = (PI * jnp.array(fb) * M_s) ** (2.0 / 3.0)
    psi_QM_b = get_qm_phase_correction(f_Ms_b, theta_int)
    psi_SS_b = get_spin_phase_correction(x_23_b, theta_int)
    rip_bbh_phase = rip_phase_full + np.array(psi_T_b + psi_QM_b + psi_SS_b)

    # Take diff, modulo 2*pi, with linear+const fit removed
    rip_bbh_unwrap = rip_bbh_phase
    # remove dc and linear (best alignment)
    phase_diff = rip_bbh_unwrap - lal_phi_band
    # Best-fit linear+const removal
    A = np.vstack([fb, np.ones_like(fb)]).T
    coef, *_ = np.linalg.lstsq(A, phase_diff, rcond=None)
    phase_diff_resid = phase_diff - (A @ coef)
    print(f"(3b) ripple (bbh+tidal_correction) - LAL phi(f), linear+const-removed:")
    print(f"     max abs residual:  {np.abs(phase_diff_resid).max():.3e} rad")
    print(f"     RMS residual:      {np.sqrt(np.mean(phase_diff_resid**2)):.3e} rad")
    print(f"     linear slope coef: {coef[0]:.3e} rad/Hz (=> tc-like shift)")
    print(f"     const offset:      {coef[1]:.3e} rad")
    for lo, hi in bands:
        m = (fb >= lo) & (fb < hi)
        if m.any():
            print(f"     {lo:>4}-{hi:<4} Hz: max|resid|={np.abs(phase_diff_resid[m]).max():.3e}"
                  f"  RMS={np.sqrt(np.mean(phase_diff_resid[m]**2)):.3e}")


if __name__ == "__main__":
    main()
