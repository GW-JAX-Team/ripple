"""
Compare amplitudes between Ripple and LAL NRTidalv3 for different mass ratios.
Check if amplitude or phase is the dominant source of mismatch.
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
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes

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


def get_lal_hp(m1, m2, chi, l1, l2):
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
    approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS_NRTidalv3")
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


def compute_overlap_loss(h1, h2):
    n = min(len(h1), len(h2))
    h1, h2 = h1[:n], h2[:n]
    norm1 = np.sqrt(np.sum(np.abs(h1)**2))
    norm2 = np.sqrt(np.sum(np.abs(h2)**2))
    h1n = h1 / norm1
    h2n = h2 / norm2
    corr = np.abs(np.fft.ifft(h1n * np.conj(h2n)))
    return 1.0 - np.max(corr) * n  # n factor from IFFT normalization


for Q in [0.25, 0.5, 1.0]:
    m1 = M_TOT / (1 + Q)
    m2 = M_TOT * Q / (1 + Q)
    l1, l2 = LAMBDA, LAMBDA

    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lt, dlt = lambdas_to_lambda_tildes(jnp.array([l1, l2, m1, m2]))
    params = jnp.array([Mc, eta, CHI, CHI, lt, dlt, DIST, 0.0, 0.0, IOTA])

    hp_rip, _ = gen_IMRPhenomXAS_NRTidalv3_hphc(fs, params, F_REF, use_lambda_tildes=True)
    hp_rip = np.asarray(hp_rip)
    hp_lal = get_lal_hp(m1, m2, CHI, l1, l2)

    n = min(len(hp_rip), len(hp_lal), len(f_np))
    hp_rip, hp_lal, fc = hp_rip[:n], hp_lal[:n], f_np[:n]

    amp_thr = 0.01 * np.max(np.abs(hp_lal))
    valid = (np.abs(hp_lal) > amp_thr) & (np.abs(hp_rip) > amp_thr)
    fc_v = fc[valid]

    amp_rip = np.abs(hp_rip[valid])
    amp_lal = np.abs(hp_lal[valid])
    amp_ratio = amp_rip / amp_lal

    phi_rip = np.unwrap(np.angle(hp_rip[valid]))
    phi_lal = np.unwrap(np.angle(hp_lal[valid]))
    delta_phi = phi_rip - phi_lal
    A = np.column_stack([fc_v, np.ones_like(fc_v)])
    coeffs, *_ = np.linalg.lstsq(A, delta_phi, rcond=None)
    delta_phi_resid = delta_phi - A @ coeffs

    amp_rms = np.sqrt(np.mean((amp_ratio - 1)**2))
    phi_rms = np.sqrt(np.mean(delta_phi_resid**2))

    # Mismatch contributions
    mismatch_phase = 0.5 * phi_rms**2
    mismatch_amp   = 0.5 * amp_rms**2

    overlap_loss = compute_overlap_loss(hp_rip, hp_lal)

    print(f"q={Q:.2f}  m1={m1:.3f}  m2={m2:.3f}")
    print(f"  Amplitude RMS fractional error: {amp_rms:.4e}")
    print(f"  Phase RMS residual:             {phi_rms:.4e} rad")
    print(f"  Estimated overlap loss (phase): {mismatch_phase:.4e}")
    print(f"  Estimated overlap loss (amp):   {mismatch_amp:.4e}")
    print(f"  Actual overlap loss:            {overlap_loss:.4e}")
    print(f"  Amplitude ratio (max dev from 1): {np.max(np.abs(amp_ratio-1)):.4e}")
    print(f"  Peak amplitude ratio location: {fc_v[np.argmax(np.abs(amp_ratio-1))]:.1f} Hz")
    print()
