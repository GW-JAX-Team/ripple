"""
Direct overlap computation matching the test_lal_overlap.py methodology.
Uses the MSA (Maximum Signal Approximation) over tc and phic.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

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
n_freq = len(f_np)

# Load PSD
psd_path = Path(__file__).parent.parent / "psds" / "ET_D_psd.txt"
psd_freqs, psd_vals = np.loadtxt(str(psd_path), unpack=True)
psd_interp = jnp.array(np.interp(f_np, psd_freqs, psd_vals))


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
    n = min(len(arr), n_freq)
    return jnp.array(arr[:n])


def noise_weighted_inner(h1, h2, S):
    n = min(len(h1), len(h2), len(S))
    return jnp.sum(h1[:n] * jnp.conj(h2[:n]) / S[:n])


def overlap_loss_msa(h1, h2, S):
    """Maximum signal approximation overlap loss, matching test_lal_overlap.py."""
    n = min(len(h1), len(h2), len(S))
    h1, h2, S = h1[:n], h2[:n], S[:n]
    norm1 = jnp.sqrt(jnp.abs(noise_weighted_inner(h1, h1, S)).real)
    norm2 = jnp.sqrt(jnp.abs(noise_weighted_inner(h2, h2, S)).real)
    h1n = h1 / norm1
    h2n = h2 / norm2
    # Maximize over tc via IFFT
    cross = h1n * jnp.conj(h2n) / S
    overlap_series = jnp.abs(jnp.fft.ifft(cross)) * n
    max_overlap = jnp.max(overlap_series)
    return 1.0 - max_overlap


for Q in [0.25, 0.35, 0.50, 0.65, 0.80, 1.0]:
    m1 = M_TOT / (1 + Q)
    m2 = M_TOT * Q / (1 + Q)
    l1, l2 = LAMBDA, LAMBDA

    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lt, dlt = lambdas_to_lambda_tildes(jnp.array([l1, l2, m1, m2]))
    params = jnp.array([Mc, eta, CHI, CHI, lt, dlt, DIST, 0.0, 0.0, IOTA])

    hp_rip, _ = gen_IMRPhenomXAS_NRTidalv3_hphc(fs, params, F_REF, use_lambda_tildes=True)
    n_rip = len(hp_rip)
    hp_rip_padded = jnp.concatenate([hp_rip, jnp.zeros(n_freq - n_rip)]) if n_rip < n_freq else hp_rip[:n_freq]

    hp_lal = get_lal_hp(m1, m2, CHI, l1, l2)
    n_lal = len(hp_lal)
    hp_lal_padded = jnp.concatenate([hp_lal, jnp.zeros(n_freq - n_lal)]) if n_lal < n_freq else hp_lal[:n_freq]

    ol = float(overlap_loss_msa(hp_rip_padded, hp_lal_padded, psd_interp))
    print(f"q={Q:.2f}  m1={m1:.3f}  m2={m2:.3f}  overlap_loss={ol:.4e}")
