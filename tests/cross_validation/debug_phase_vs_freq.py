"""
Localize the M-dependent phase residual in frequency.

For each total mass, this script:
1. Generates LAL and Ripple waveforms
2. Computes phase(f) for both (unwrapped)
3. Fits and removes a linear+constant in f (time+phase offset)
4. Shows the residual as a function of f for each mass
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import jax
import jax.numpy as jnp
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes

MTSUN = 4.925491025543576e-6  # seconds

# Fixed parameters
CHI1 = 0.0
CHI2 = 0.0
LAMBDA1 = 400.0
LAMBDA2 = 400.0
Q = 1.0  # mass ratio m1/m2
D_L = 100.0  # Mpc
TC = 0.0
PHI_C = 0.0
F_REF = 20.0
F_LOW = 20.0
F_HIGH = 2048.0
DF = 1.0 / 32.0

masses_total = [1.4, 2.0, 2.8, 3.5, 4.5, 5.5]  # solar masses

approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS_NRTidalv3")


def get_lal_hf(m1, m2, chi1, chi2, lambda1, lambda2, d_L, phi_c,
               f_low, f_high, df, f_ref):
    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    distance = d_L * 1e6 * lal.PC_SI
    inclination = 0.0

    laldict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, lambda1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, lambda2)
    quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(lambda1)
    quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(lambda2)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)

    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_kg, m2_kg,
        0.0, 0.0, chi1,
        0.0, 0.0, chi2,
        distance, inclination, phi_c,
        0, 0, 0,
        df, f_low, f_high, f_ref,
        laldict, approximant,
    )

    freqs_lal = np.arange(len(hp.data.data)) * df
    mask = (freqs_lal > f_low) & (freqs_lal < f_high)
    return freqs_lal[mask], hp.data.data[mask]


def get_ripple_hf(m1, m2, chi1, chi2, lambda1, lambda2, d_L, tc, phi_c, freqs, f_ref):
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lamt, dlamt = lambdas_to_lambda_tildes(jnp.array([lambda1, lambda2, m1, m2]))
    params = jnp.array([Mc, eta, chi1, chi2, lamt, dlamt, d_L, tc, phi_c])
    h = gen_IMRPhenomXAS_NRTidalv3(freqs, params, f_ref)
    return np.array(h)


def ol_direct(h1, h2, df):
    """Direct overlap loss 1 - Re(<h1|h2>)/sqrt(<h1|h1><h2|h2>) with flat PSD."""
    inner12 = np.sum(np.conj(h1) * h2) * df
    inner11 = np.sum(np.abs(h1)**2) * df
    inner22 = np.sum(np.abs(h2)**2) * df
    return 1.0 - np.real(inner12) / np.sqrt(inner11 * inner22)


def amp_weighted_rms(dphi, amp, mask):
    if not np.any(mask):
        return 0.0
    w = amp[mask]**2
    return np.sqrt(np.sum(w * dphi[mask]**2) / np.sum(w))


def main():
    print(f"{'M_tot':>8} {'OL_direct':>12} {'RMS_unwtd':>12} {'max_res':>12} {'f@max_res':>10}")
    print("-" * 65)

    results = {}
    for M_tot in masses_total:
        m1 = M_tot * Q / (1 + Q)
        m2 = M_tot / (1 + Q)

        freqs_lal, h_lal = get_lal_hf(
            m1, m2, CHI1, CHI2, LAMBDA1, LAMBDA2,
            D_L, PHI_C, F_LOW, F_HIGH, DF, F_REF,
        )

        h_ripple = get_ripple_hf(
            m1, m2, CHI1, CHI2, LAMBDA1, LAMBDA2,
            D_L, TC, PHI_C, jnp.array(freqs_lal), F_REF,
        )

        ol = ol_direct(h_lal, h_ripple, DF)

        # Remove zero-amplitude points for phase analysis
        good = (np.abs(h_lal) > 0) & (np.abs(h_ripple) > 0)
        freqs = freqs_lal[good]
        h_lal_g = h_lal[good]
        h_rip_g = h_ripple[good]

        # Phase difference, remove linear+constant
        phi_lal = np.unwrap(np.angle(h_lal_g))
        phi_rip = np.unwrap(np.angle(h_rip_g))
        dphi = phi_lal - phi_rip
        p = np.polyfit(freqs, dphi, 1)
        dphi_res = dphi - np.polyval(p, freqs)

        rms_res = np.sqrt(np.mean(dphi_res**2))
        max_res = np.max(np.abs(dphi_res))
        f_at_max = freqs[np.argmax(np.abs(dphi_res))]
        amp = np.abs(h_rip_g)

        print(f"{M_tot:8.2f} {ol:12.3e} {rms_res:12.3e} {max_res:12.3e} {f_at_max:10.1f}")

        results[M_tot] = {
            "freqs": freqs,
            "dphi_res": dphi_res,
            "amp": amp,
            "ol": ol,
        }

    # Frequency-band breakdown — unweighted and amplitude-weighted RMS
    print()
    print("Unweighted RMS phase residual by band:")
    print(f"{'M_tot':>8} {'<200Hz':>12} {'200-500Hz':>12} {'>500Hz':>12}")
    print("-" * 50)
    for M_tot, r in results.items():
        freqs = r["freqs"]
        dphi_res = r["dphi_res"]
        b1 = freqs < 200
        b2 = (freqs >= 200) & (freqs < 500)
        b3 = freqs >= 500
        rms = lambda m: np.sqrt(np.mean(dphi_res[m]**2)) if np.any(m) else 0.0
        print(f"{M_tot:8.2f} {rms(b1):12.3e} {rms(b2):12.3e} {rms(b3):12.3e}")

    print()
    print("Amplitude-weighted RMS phase residual by band:")
    print(f"{'M_tot':>8} {'<200Hz':>12} {'200-500Hz':>12} {'>500Hz':>12} {'OL_direct':>12}")
    print("-" * 64)
    for M_tot, r in results.items():
        freqs = r["freqs"]
        dphi_res = r["dphi_res"]
        amp = r["amp"]
        b1 = freqs < 200
        b2 = (freqs >= 200) & (freqs < 500)
        b3 = freqs >= 500
        rw = lambda m: amp_weighted_rms(dphi_res, amp, m)
        print(f"{M_tot:8.2f} {rw(b1):12.3e} {rw(b2):12.3e} {rw(b3):12.3e} {r['ol']:12.3e}")

    # Power-law fits (inspiral-only masses)
    M_lo = np.array([M for M in sorted(results.keys()) if M <= 2.8])
    if len(M_lo) >= 2:
        rms_lo = np.array([np.sqrt(np.mean(results[M]["dphi_res"]**2)) for M in M_lo])
        ol_lo = np.array([results[M]["ol"] for M in M_lo])
        slope_rms, _ = np.polyfit(np.log10(M_lo), np.log10(rms_lo), 1)
        slope_ol, _ = np.polyfit(np.log10(M_lo), np.log10(ol_lo), 1)
        print(f"\nPower law (M ≤ 2.8): RMS ∝ M^{slope_rms:.2f},  OL ∝ M^{slope_ol:.2f}")

    M_all = np.array(sorted(results.keys()))
    ol_all = np.array([results[M]["ol"] for M in M_all])
    slope_ol_all, _ = np.polyfit(np.log10(M_all), np.log10(ol_all), 1)
    print(f"Power law (all M):   OL ∝ M^{slope_ol_all:.2f}")


if __name__ == "__main__":
    main()
