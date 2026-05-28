"""
Reproduce exact test overlap computation and diagnose source of mismatch.
Compare the overlap with what we get from just the linear phase residual.
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import compute_overlap_loss as official_overlap_loss, get_nyquist_mask

from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes

# Use the WORST-CASE sample from the CSV
# m1=2.717716, m2=0.563548, lambda1=961.445094, lambda2=4871.974038, overlap_loss=8.85e-7
M1 = 2.717716
M2 = 0.563548
L1 = 961.445094
L2 = 4871.974038
CHI1 = 0.025087
CHI2 = 0.012289
DIST = 102.550476
TC = 0.0
PHIC = 0.081693
IOTA = 2.188593

# Also test with our simple case
CASES = [
    {"m1": M1, "m2": M2, "l1": L1, "l2": L2, "chi1": CHI1, "chi2": CHI2,
     "dist": DIST, "tc": TC, "phic": PHIC, "iota": IOTA, "label": "worst-case"},
    {"m1": 1.6, "m2": 0.4, "l1": 400.0, "l2": 400.0, "chi1": 0.0, "chi2": 0.0,
     "dist": 100.0, "tc": 0.0, "phic": 0.0, "iota": 0.4, "label": "q=0.25,L=400"},
]

F_L = 20.0
F_U = 4096.0
T = 32.0
DF = 1.0 / T
F_REF = 20.0

fs = jnp.arange(F_L, F_U, DF)
f_np = np.asarray(fs)
n_freq = len(f_np)
nyquist_mask = get_nyquist_mask(fs)

psd_path = Path(__file__).parent.parent / "psds" / "ET_D_psd.txt"
psd_freqs, psd_vals = np.loadtxt(str(psd_path), unpack=True)
psd_interp = jnp.array(np.interp(f_np, psd_freqs, psd_vals))


def get_lal_hp(m1, m2, chi1, chi2, l1, l2, dist, iota, tc, phic):
    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    dist_m = dist * 1e6 * lal.PC_SI
    laldict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, l1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, l2)
    q1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1)
    q2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, q1 - 1)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, q2 - 1)
    approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS_NRTidalv3")
    phi0_lal = phic / 2.0  # LAL phi0 = phic/2
    hp, _ = lalsim.SimInspiralChooseFDWaveform(
        m1_kg, m2_kg, 0, 0, chi1, 0, 0, chi2,
        dist_m, iota, phi0_lal, 0, 0, 0,
        DF, F_L, F_U, F_REF, laldict, approx,
    )
    # Apply tc as a phase rotation in frequency domain
    freq_arr = np.arange(len(hp.data.data)) * DF
    mask = (freq_arr >= F_L) & (freq_arr < F_U)
    arr = hp.data.data[mask].copy()
    if tc != 0.0:
        arr *= np.exp(-2j * np.pi * freq_arr[mask] * tc)
    n = min(len(arr), n_freq)
    result = np.zeros(n_freq, dtype=complex)
    result[:n] = arr[:n]
    return jnp.array(result)


for case in CASES:
    m1, m2 = case["m1"], case["m2"]
    l1, l2 = case["l1"], case["l2"]
    chi1, chi2 = case["chi1"], case["chi2"]
    dist, tc, phic, iota = case["dist"], case["tc"], case["phic"], case["iota"]
    label = case["label"]

    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lt, dlt = lambdas_to_lambda_tildes(jnp.array([l1, l2, m1, m2]))
    params = jnp.array([Mc, eta, chi1, chi2, lt, dlt, dist, tc, phic, iota])

    hp_rip, _ = gen_IMRPhenomXAS_NRTidalv3_hphc(fs, params, F_REF, use_lambda_tildes=True)
    hp_rip_masked = hp_rip * nyquist_mask

    hp_lal = get_lal_hp(m1, m2, chi1, chi2, l1, l2, dist, iota, tc, phic)
    hp_lal_masked = hp_lal * nyquist_mask

    ol = float(official_overlap_loss(hp_rip_masked, hp_lal_masked, psd_interp, fs))
    print(f"\n=== {label} ===")
    print(f"  m1={m1:.3f}, m2={m2:.3f}, q={m2/m1:.3f}")
    print(f"  Overlap loss: {ol:.4e}")

    # Phase residual
    n = n_freq
    amp_thr = 0.01 * float(jnp.max(jnp.abs(hp_lal_masked)))
    valid = (jnp.abs(hp_lal_masked) > amp_thr) & (jnp.abs(hp_rip_masked) > amp_thr)
    valid = np.asarray(valid)

    hp_rip_v = np.asarray(hp_rip_masked)[valid]
    hp_lal_v = np.asarray(hp_lal_masked)[valid]
    fv = f_np[valid]

    phi_rip = np.unwrap(np.angle(hp_rip_v))
    phi_lal = np.unwrap(np.angle(hp_lal_v))
    delta_phi = phi_rip - phi_lal

    # No detrending
    phi_rms_no_detrend = np.sqrt(np.mean(delta_phi**2))

    # Remove constant only (phic)
    delta_phi_no_const = delta_phi - np.mean(delta_phi)
    phi_rms_no_const = np.sqrt(np.mean(delta_phi_no_const**2))

    # Remove linear+constant (tc + phic)
    A_fit = np.column_stack([fv, np.ones_like(fv)])
    coeffs, *_ = np.linalg.lstsq(A_fit, delta_phi, rcond=None)
    delta_phi_no_linear = delta_phi - A_fit @ coeffs
    phi_rms_no_linear = np.sqrt(np.mean(delta_phi_no_linear**2))

    print(f"  Phase residual:")
    print(f"    No detrending:     {phi_rms_no_detrend:.4e} rad  -> mismatch ~{0.5*phi_rms_no_detrend**2:.4e}")
    print(f"    Remove const only: {phi_rms_no_const:.4e} rad  -> mismatch ~{0.5*phi_rms_no_const**2:.4e}")
    print(f"    Remove linear:     {phi_rms_no_linear:.4e} rad  -> mismatch ~{0.5*phi_rms_no_linear**2:.4e}")

    # Noise-weighted residual
    psd_v = np.interp(fv, psd_freqs, psd_vals)
    amp_sq = np.abs(hp_lal_v)**2

    weight = amp_sq / psd_v
    nw_phi_rms_no_linear = np.sqrt(np.average(delta_phi_no_linear**2, weights=weight))
    print(f"    NW remove linear:  {nw_phi_rms_no_linear:.4e} rad  -> mismatch ~{0.5*nw_phi_rms_no_linear**2:.4e}")

    # Print the phase difference at a few frequencies
    low_f_mask = fv < 100
    if np.any(low_f_mask):
        print(f"  Phase diff at f<100Hz: mean={np.mean(delta_phi[low_f_mask]):.4e}, std={np.std(delta_phi[low_f_mask]):.4e}")
    mid_f_mask = (fv >= 100) & (fv < 500)
    if np.any(mid_f_mask):
        print(f"  Phase diff 100-500Hz: mean={np.mean(delta_phi[mid_f_mask]):.4e}, std={np.std(delta_phi[mid_f_mask]):.4e}")
    high_f_mask = fv >= 500
    if np.any(high_f_mask):
        print(f"  Phase diff f>500Hz: mean={np.mean(delta_phi[high_f_mask]):.4e}, std={np.std(delta_phi[high_f_mask]):.4e}")
