"""
Phase-level diagnostic for IMRPhenomXAS_NRTidalv3 mass-ratio dependence.

Fixes M_total = 2.0 M_sun, Lambda_1 = Lambda_2 = 400, and scans mass ratio
q = m2/m1 from 0.25 to 1.0. For each system the unwrapped phase of h_+ is
extracted from both LAL and Ripple, the best-fit linear+constant (tc/phic) is
removed, and the amplitude-masked residual is plotted.

A separate panel shows the noise-weighted RMS phase residual vs q.

Usage:
    python diagnose_nrtidal_phase.py

Output:
    diagnose_nrtidal_phase.png
"""

import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

try:
    import lal
    import lalsimulation as lalsim
except ImportError:
    sys.exit("LALSuite not available.")

from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes

# ── frequency grid ──────────────────────────────────────────────────────────
F_L   = 20.0
F_U   = 4096.0
T     = 32.0
DF    = 1.0 / T
F_REF = 20.0
IOTA  = 0.4       # off-axis so hp amplitude doesn't vanish

fs    = jnp.arange(F_L, F_U, DF)
f_np  = np.asarray(fs)

# ── PSD for noise-weighting ──────────────────────────────────────────────────
from pathlib import Path
psd_path = Path(__file__).parent.parent / "psds" / "ET_D_psd.txt"
psd_freqs, psd_vals = np.loadtxt(str(psd_path), unpack=True)
psd_interp = np.interp(f_np, psd_freqs, psd_vals)

# ── parameter grid: fixed M_tot=2.0, vary q ─────────────────────────────────
M_TOT   = 2.0
LAMBDA  = 400.0
CHI     = 0.0
DIST    = 100.0

# q = m2/m1, m1 >= m2 => m1 = M_TOT/(1+q), m2 = M_TOT*q/(1+q)
Q_VALUES = [0.25, 0.35, 0.50, 0.65, 0.80, 1.00]


def masses(q):
    m1 = M_TOT / (1 + q)
    m2 = M_TOT * q / (1 + q)
    return m1, m2


def get_lal_hp(m1, m2, chi1, chi2, l1, l2):
    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    dist  = DIST * 1e6 * lal.PC_SI
    laldict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, l1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, l2)
    q1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1)
    q2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, q1 - 1)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, q2 - 1)
    approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS_NRTidalv3")
    hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
        m1_kg, m2_kg, 0.0, 0.0, chi1, 0.0, 0.0, chi2,
        dist, IOTA, 0.0, 0, 0, 0,
        DF, F_L, F_U, F_REF, laldict, approx,
    )
    freqs_lal = np.arange(len(hp_lal.data.data)) * DF
    mask = (freqs_lal > F_L) & (freqs_lal < F_U)
    return np.asarray(hp_lal.data.data[mask])


def get_ripple_hp(m1, m2, chi1, chi2, l1, l2):
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lt, dlt = lambdas_to_lambda_tildes(jnp.array([l1, l2, m1, m2]))
    params  = jnp.array([Mc, eta, chi1, chi2, lt, dlt, DIST, 0.0, 0.0, IOTA])
    hp, _   = gen_IMRPhenomXAS_NRTidalv3_hphc(fs, params, F_REF, use_lambda_tildes=True)
    return np.asarray(hp)


def phase_residual(hp_rip, hp_lal):
    """Unwrap both phases, subtract linear+constant, return (f, residual, amp_mask)."""
    n = min(len(hp_rip), len(hp_lal), len(f_np))
    hp_rip = hp_rip[:n]
    hp_lal = hp_lal[:n]
    fc     = f_np[:n]

    # Amplitude mask: both non-zero and above 1% of LAL peak
    amp_threshold = 0.01 * np.max(np.abs(hp_lal))
    valid = (np.abs(hp_lal) > amp_threshold) & (np.abs(hp_rip) > amp_threshold)

    phi_rip = np.unwrap(np.angle(hp_rip[valid]))
    phi_lal = np.unwrap(np.angle(hp_lal[valid]))
    fv      = fc[valid]

    delta = phi_rip - phi_lal
    # Remove best-fit linear+constant
    A      = np.column_stack([fv, np.ones_like(fv)])
    coeffs, *_ = np.linalg.lstsq(A, delta, rcond=None)
    resid  = delta - A @ coeffs
    return fv, resid, np.abs(hp_lal[valid])


def noise_weighted_rms(fv, resid, amp):
    """Noise-weighted RMS: <resid^2 * amp^2 / S(f)> / <amp^2 / S(f)>."""
    psd_v  = np.interp(fv, psd_freqs, psd_vals)
    weight = amp**2 / psd_v
    return np.sqrt(np.average(resid**2, weights=weight))


# ── run ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
nw_rms_list = []

for ax, q in zip(axes, Q_VALUES):
    m1, m2 = masses(q)
    l1, l2 = LAMBDA, LAMBDA   # equal lambdas → asymmetry is purely from mass ratio
    print(f"  q={q:.2f}  m1={m1:.3f}  m2={m2:.3f} ...", end=" ", flush=True)

    hp_lal    = get_lal_hp(   m1, m2, CHI, CHI, l1, l2)
    hp_ripple = get_ripple_hp(m1, m2, CHI, CHI, l1, l2)

    fv, resid, amp = phase_residual(hp_ripple, hp_lal)
    rms    = np.sqrt(np.mean(resid**2))
    nw_rms = noise_weighted_rms(fv, resid, amp)
    nw_rms_list.append(nw_rms)
    print(f"plain rms={rms:.2e} rad   noise-wtd rms={nw_rms:.2e} rad")

    ax.plot(fv, resid, lw=0.7)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_title(f"$q={q:.2f}$  ($m_1={m1:.2f},\\,m_2={m2:.2f}$)\n"
                 f"plain rms={rms:.1e}  nw-rms={nw_rms:.1e} rad", fontsize=9)
    ax.set_xlabel("$f$  [Hz]")
    ax.set_ylabel(r"$\Delta\phi_{\rm res}$  [rad]")
    ax.set_xlim(F_L, 2000)
    ax.grid(ls="--", alpha=0.4)

fig.suptitle(
    "IMRPhenomXAS_NRTidalv3: Ripple − LAL phase residual (linear drift removed)\n"
    r"Fixed $M_{\rm tot}=2.0\,M_\odot$, $\Lambda_1=\Lambda_2=400$, $\chi=0$",
    fontsize=11,
)
fig.tight_layout()
outfile = "diagnose_nrtidal_phase.png"
fig.savefig(outfile, dpi=150)
print(f"\nPlot saved to {outfile}")
plt.show()
