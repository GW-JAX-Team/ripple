"""
Compare BBH (IMRPhenomXAS) phase and full tidal phase for different mass ratios.
This isolates whether the mismatch is in the BBH part or the tidal part.
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

from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.constants import MTSUN, PI

M_TOT = 2.0
LAMBDA = 400.0
CHI = 0.0
DIST = 100.0
IOTA = 0.4
F_L = 20.0
F_U = 4096.0
T = 32.0
DF = 1.0 / T
F_REF = 20.0

fs = jnp.arange(F_L, F_U, DF)
f_np = np.asarray(fs)


def get_lal_hp(m1, m2, chi, l1, l2, approx_str):
    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    dist = DIST * 1e6 * lal.PC_SI
    laldict = lal.CreateDict()
    if l1 > 0 or l2 > 0:
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
    arr = hp.data.data[mask]
    n = min(len(arr), len(f_np))
    return arr[:n]


def phase_rms(hp1, hp2, f_arr):
    n = min(len(hp1), len(hp2), len(f_arr))
    hp1, hp2, fc = hp1[:n], hp2[:n], f_arr[:n]
    amp_thr = 0.01 * max(np.max(np.abs(hp1)), np.max(np.abs(hp2)))
    valid = (np.abs(hp1) > amp_thr) & (np.abs(hp2) > amp_thr)
    phi1 = np.unwrap(np.angle(hp1[valid]))
    phi2 = np.unwrap(np.angle(hp2[valid]))
    delta = phi1 - phi2
    fv = fc[valid]
    A = np.column_stack([fv, np.ones_like(fv)])
    coeffs, *_ = np.linalg.lstsq(A, delta, rcond=None)
    resid = delta - A @ coeffs
    return np.sqrt(np.mean(resid**2))


for Q in [0.25, 0.5, 1.0]:
    m1 = M_TOT / (1 + Q)
    m2 = M_TOT * Q / (1 + Q)
    l1, l2 = LAMBDA, LAMBDA

    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lt, dlt = lambdas_to_lambda_tildes(jnp.array([l1, l2, m1, m2]))
    params_tidal = jnp.array([Mc, eta, CHI, CHI, lt, dlt, DIST, 0.0, 0.0, IOTA])
    params_bbh = jnp.array([Mc, eta, CHI, CHI, DIST, 0.0, 0.0, IOTA])

    # Ripple waveforms
    hp_rip_tidal, _ = gen_IMRPhenomXAS_NRTidalv3_hphc(fs, params_tidal, F_REF, use_lambda_tildes=True)
    hp_rip_bbh,   _ = gen_IMRPhenomXAS_hphc(fs, params_bbh, F_REF)

    hp_rip_tidal = np.asarray(hp_rip_tidal)
    hp_rip_bbh   = np.asarray(hp_rip_bbh)

    # LAL waveforms
    hp_lal_tidal = get_lal_hp(m1, m2, CHI, l1, l2, "IMRPhenomXAS_NRTidalv3")
    hp_lal_bbh   = get_lal_hp(m1, m2, CHI, 0.0, 0.0, "IMRPhenomXAS")

    # Phase residuals (after removing linear+constant)
    rms_tidal = phase_rms(hp_rip_tidal, hp_lal_tidal, f_np)
    rms_bbh   = phase_rms(hp_rip_bbh,   hp_lal_bbh,   f_np)

    print(f"q={Q:.2f}  m1={m1:.3f}  m2={m2:.3f}")
    print(f"  BBH   phase RMS residual: {rms_bbh:.4e} rad")
    print(f"  Tidal phase RMS residual: {rms_tidal:.4e} rad")
    print()
