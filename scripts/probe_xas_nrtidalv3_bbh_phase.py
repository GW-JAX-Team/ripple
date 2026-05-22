"""Stage-2 probe: compare ripple's "phi(f) without tidal subtraction" against LAL.

Builds ripple's (bbh_psi + phase_shift) by reproducing the
_gen_IMRPhenomXAS_NRTidalv3 code path through the bbh side, then compares to
LAL's PhenomXOnlyReturnPhase output sample-by-sample.

This isolates the BBH-phase / linb-alignment / phifRef portion from the
tidal correction (which we already verified matches to ~1e-12).
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
from ripplegw.waveforms.NRTidalv3_utils import (
    _get_merger_frequency,
    general_planck_taper,
    phenomx_tidal_phase,
    phenomx_tidal_phase_derivative,
)
from ripplegw.waveforms import IMRPhenomX_utils
from ripplegw.waveforms.IMRPhenomXAS import Phase, PhaseDerivative


def main():
    data = np.load("tests/cross_validation/lal_cache/IMRPhenomXAS_NRTidalv3_T128.npz",
                   allow_pickle=False)
    i = 1
    m1, m2, s1z, s2z, l1, l2, dL, tc, phic, inc = data["theta_batch"][i]
    M = m1 + m2
    M_s = M * MTSUN
    theta_int = jnp.array([m1, m2, s1z, s2z, l1, l2])
    theta_bbh = jnp.array([m1, m2, s1z, s2z])
    print(f"Sample idx=1: m1={m1:.4f} m2={m2:.4f} s1z={s1z:.4f} s2z={s2z:.4f}"
          f" l1={l1:.1f} l2={l2:.1f}")

    # Frequency grid: match the test (T=128, df=1/128, f in (20, 4096))
    T = 128.0
    df = 1.0 / T
    fs = np.arange(20.0, 4096.0, df)
    f_ref = 20.0
    f_l, f_u = 20.0, 4096.0

    # ----- LAL: PhenomXOnlyReturnPhase phi(f) -----
    p_dict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXOnlyReturnPhase(p_dict, 1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(p_dict, float(l1))
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(p_dict, float(l2))
    lalsim.SimInspiralWaveformParamsInsertPhenomXTidalFlag(p_dict, 3)  # NRTidalv3
    m1_SI = float(m1) * lal.MSUN_SI
    m2_SI = float(m2) * lal.MSUN_SI
    hp, _ = lalsim.SimInspiralChooseFDWaveform(
        m1_SI, m2_SI, 0.0, 0.0, float(s1z), 0.0, 0.0, float(s2z),
        float(dL) * 1e6 * lal.PC_SI, 0.0, 0.0, 0.0, 0.0, 0.0,
        df, f_l, f_u, f_ref, p_dict, lalsim.IMRPhenomXAS_NRTidalv3,
    )
    lal_freqs = np.arange(hp.data.length) * hp.deltaF
    band = (lal_freqs >= f_l) & (lal_freqs < f_u)
    lal_phi = np.asarray(hp.data.data)[band].real
    lal_band_freqs = lal_freqs[band]
    # Align ripple grid to LAL band exactly
    fb = lal_band_freqs.astype(np.float64)

    # ----- ripple: rebuild (bbh_psi + phase_shift) directly -----
    f_merger = float(_get_merger_frequency(theta_int))
    df_ripple = df  # same as LAL deltaF
    # f_final mode = current default "plus_df"
    f_final_raw = float(fb[-1]) + df_ripple
    f_final = min(f_merger, f_final_raw)

    P_P_fref = float(general_planck_taper(
        jnp.array(f_ref * M_s), 1.15 * f_merger * M_s, 1.35 * f_merger * M_s
    ))

    # dphiT analytic (current default for DPHIT)
    dphiT = float(phenomx_tidal_phase_derivative(theta_int, f_final * M_s))

    # dphiXAS secant (current default)
    Mf_final = f_final * M_s
    bbh_phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table
    Phase_f_final = float(Phase(jnp.array(f_final), theta_bbh, bbh_phase_coeffs))
    Phase_f_final_minus = float(Phase(jnp.array(f_final - df_ripple), theta_bbh, bbh_phase_coeffs))
    dphiXAS_secant = (Phase_f_final - Phase_f_final_minus) / (df_ripple * M_s)
    dphiXAS_analytic = float(PhaseDerivative(jnp.array(f_final), theta_bbh, bbh_phase_coeffs)) / M_s
    print(f"\nf_merger      = {f_merger:.6f} Hz")
    print(f"f_final       = {f_final:.6f} Hz  (=f_merger? {abs(f_final-f_merger)<1e-12})")
    print(f"dphiT  (analytic) = {dphiT:.12e}")
    print(f"dphiXAS (secant)  = {dphiXAS_secant:.12e}")
    print(f"dphiXAS (analytic)= {dphiXAS_analytic:.12e}")
    print(f"  delta secant-analytic = {dphiXAS_secant - dphiXAS_analytic:.3e}")

    linb = dphiT - dphiXAS_secant  # using current "secant" default

    phiTfRef = float(phenomx_tidal_phase(theta_int, f_ref * M_s))
    Phase_fref = float(Phase(jnp.array(f_ref), theta_bbh, bbh_phase_coeffs))
    # ripple's bbh_psi(f) is just Phase(f)
    bbh_psi = np.array(Phase(jnp.array(fb), theta_bbh, bbh_phase_coeffs))

    f_Ms = fb * M_s
    f_ref_M_s = f_ref * M_s
    # ext_phase_contrib = 0 here (tc=0, phic=0)
    phase_shift = (
        linb * (f_Ms - f_ref_M_s)
        - Phase_fref
        + phiTfRef
        + PI / 4.0
        - PI
    )
    ripple_phi = bbh_psi + phase_shift  # equivalent to LAL's phi(f) modulo conv.

    # ----- Compare -----
    # Caveat: there can be an overall constant phase difference due to phi0
    # convention. Remove it; what we really want is to see if there's a
    # nonlinear-in-f residual.
    diff = ripple_phi - lal_phi
    # remove const + linear-in-f best fit
    A = np.vstack([fb, np.ones_like(fb)]).T
    coef, *_ = np.linalg.lstsq(A, diff, rcond=None)
    resid = diff - A @ coef
    print(f"\nripple_phi - lal_phi (linear+const removed):")
    print(f"  best-fit slope: {coef[0]:.6e} rad/Hz  (~ tshift = {coef[0]/(2*PI):.3e} s)")
    print(f"  best-fit const: {coef[1]:.6e} rad")
    print(f"  residual: max|d|={np.abs(resid).max():.3e}"
          f"  RMS={np.sqrt(np.mean(resid**2)):.3e} rad")
    # Resolve by frequency band
    bands = [(20, 50), (50, 100), (100, 200), (200, 500), (500, 1000),
             (1000, 2000), (2000, 4000)]
    for lo, hi in bands:
        m = (fb >= lo) & (fb < hi)
        if m.any():
            print(f"    {lo:>4}-{hi:<4} Hz: max|resid|={np.abs(resid[m]).max():.3e}"
                  f"  RMS={np.sqrt(np.mean(resid[m]**2)):.3e}")

    # Also test linb-analytic variant to see if it gives a different residual shape
    linb_an = dphiT - dphiXAS_analytic
    phase_shift_an = (
        linb_an * (f_Ms - f_ref_M_s) - Phase_fref + phiTfRef + PI / 4.0 - PI
    )
    rphi_an = bbh_psi + phase_shift_an
    diff_an = rphi_an - lal_phi
    coef_an, *_ = np.linalg.lstsq(A, diff_an, rcond=None)
    resid_an = diff_an - A @ coef_an
    print(f"\n[ANALYTIC dphiXAS variant]")
    print(f"  residual: max|d|={np.abs(resid_an).max():.3e}"
          f"  RMS={np.sqrt(np.mean(resid_an**2)):.3e}")


if __name__ == "__main__":
    main()
