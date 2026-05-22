#!/usr/bin/env python
"""Direct comparison of ripple's PhaseDerivative vs LAL's IMRPhenomX_dPhase_22.

Also tests the exact phiTfRef and psi_T construction to find the discrepancy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.waveforms.IMRPhenomXAS import (
    Phase as RipplePhase,
    PhaseDerivative as RipplePhaseDeriv,
)
from ripplegw.waveforms.IMRPhenomX_utils import PhenomX_phase_coeff_table
from ripplegw.waveforms.NRTidalv3_utils import (
    phenomx_tidal_phase,
    phenomx_tidal_phase_derivative,
    fullTidalPhaseCorrection,
    general_planck_taper,
    _get_merger_frequency,
    get_tidal_phase,
    get_tidal_phase_PN,
    get_tidalphasePN_coeffs,
    get_NRTidalv3_coefficients,
)
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import get_qm_phase_correction, get_spin_phase_correction
from ripplegw.constants import MTSUN, PI

# ─── Test parameters ─────────────────────────────────────────────────────────
m1, m2 = 1.4, 1.35
chi1, chi2 = 0.02, 0.015
lambda1, lambda2 = 400.0, 300.0
f_ref = 20.0
T = 128.0
df = 1.0 / T

Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
M_s = (m1 + m2) * MTSUN
bbh_phase_coeffs = PhenomX_phase_coeff_table

# ─── Test 1: PhaseDerivative comparison at multiple frequencies ──────────────

print("=" * 80)
print("Test 1: PhaseDerivative at multiple frequencies")
print("=" * 80)

# We can't directly call LAL's IMRPhenomX_dPhase_22 from Python, but we can
# compute the derivative of LAL's XAS waveform phase and compare with ripple.

# Generate LAL XAS waveform
m1_kg = m1 * lal.MSUN_SI
m2_kg = m2 * lal.MSUN_SI
dist_SI = 100.0 * 1e6 * lal.PC_SI
approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")

hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0.0, 0.0, chi1, 0.0, 0.0, chi2,
    dist_SI, 0.0, 0.0, 0, 0, 0,
    df, 20.0, 4096.0, f_ref, None, approximant,
)
hp_lal_data = np.array(hp_lal.data.data)
f_lal = np.arange(len(hp_lal_data)) * df

# Trim to nonzero
nonzero = np.where(np.abs(hp_lal_data) > 0)[0]
hp_lal_data = hp_lal_data[nonzero[0]:nonzero[-1]+1]
f_lal = f_lal[nonzero[0]:nonzero[-1]+1]

# Compute LAL phase derivative using 5-point stencil
phase_lal = np.unwrap(np.angle(hp_lal_data))
dphi_lal_5pt = np.zeros_like(phase_lal)
for i in range(2, len(phase_lal) - 2):
    dphi_lal_5pt[i] = (phase_lal[i-2] - 8*phase_lal[i-1] + 8*phase_lal[i+1] - phase_lal[i+2]) / (12 * df)

# Define theta_bbh for ripple phase derivative
theta_bbh = jnp.array([m1, m2, chi1, chi2])

# Compute ripple phase derivative at a subset of frequencies
f_test = np.concatenate([
    np.linspace(20, 500, 20),
    np.linspace(500, 2037, 30),
])

dphi_rip_test = np.array([
    float(RipplePhaseDeriv(float(f_val), theta_bbh, bbh_phase_coeffs) / M_s)
    for f_val in f_test
])

# Compare at selected frequencies
print(f"\n{'f(Hz)':>8} {'LAL dphi/df':>16} {'Ripple dphi/df':>16} {'Difference':>14} {'Rel diff':>12}")
for f_check in [20, 50, 100, 200, 500, 1000, 1500, 2000, 2037]:
    idx = np.argmin(np.abs(f_lal - f_check))
    # Interpolate ripple derivative at this frequency
    rip_idx = np.argmin(np.abs(f_test - f_lal[idx]))
    if 2 <= idx < len(dphi_lal_5pt) - 2:
        lal_val = dphi_lal_5pt[idx]
        rip_val = float(dphi_rip_test[rip_idx])
        diff = lal_val - rip_val
        rel = abs(diff / lal_val) if abs(lal_val) > 1e-10 else float('nan')
        print(f"{f_lal[idx]:8.2f} {lal_val:+16.6e} {rip_val:+16.6e} {diff:+14.6e} {rel:12.4e}")

# ─── Test 2: phiTfRef consistency ────────────────────────────────────────────

print("\n" + "=" * 80)
print("Test 2: phiTfRef vs psi_T at f_ref")
print("=" * 80)

f_ref_Ms = f_ref * M_s
f_merger = float(_get_merger_frequency(theta_intrinsic))

# phiTfRef from phenomx_tidal_phase (used when tapering)
phiTfRef_taper = float(phenomx_tidal_phase(theta_intrinsic, f_ref_Ms))
print(f"phiTfRef (phenomx_tidal_phase): {phiTfRef_taper:.10f}")

# phiTfRef from fullTidalPhaseCorrection (used when no_taper)
P_P_fref = general_planck_taper(f_ref_Ms, 1.15 * f_merger * M_s, 1.35 * f_merger * M_s)
phiTfRef_notaper = float(fullTidalPhaseCorrection(f_ref_Ms, theta_intrinsic, jnp.array(P_P_fref)))
print(f"phiTfRef (fullTidalPhaseCorrection, P_P={P_P_fref:.6f}): {phiTfRef_notaper:.10f}")

# psi_T at f_ref (without taper)
PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)
x_ref = PI * f_ref_Ms
NRTidalv3_phase_ref = float(get_tidal_phase(x_ref, NRTidalv3_coeffs, PN_coeffs))
print(f"\nNRTidalv3_phase at f_ref: {NRTidalv3_phase_ref:.10f}")

# psi_QM and psi_SS at f_ref
psi_QM_ref = float(get_qm_phase_correction(f_ref_Ms, theta_intrinsic))
psi_SS_ref = float(get_spin_phase_correction(x_ref ** (2.0/3.0), theta_intrinsic))
print(f"psi_QM at f_ref: {psi_QM_ref:.10f}")
print(f"psi_SS at f_ref: {psi_SS_ref:.10f}")

# psi_T at f_ref (no taper applied at f_ref since f_ref << taper_start)
psi_T_ref_no_taper = NRTidalv3_phase_ref + psi_QM_ref + psi_SS_ref
print(f"\npsi_T + psi_QM + psi_SS at f_ref (no taper): {psi_T_ref_no_taper:.10f}")

# The key question: does phiTfRef match psi_T + psi_QM + psi_SS at f_ref?
print(f"\nphiTfRef (taper) vs psi_T+QM+SS at f_ref:")
print(f"  Difference: {phiTfRef_taper - psi_T_ref_no_taper:.10e}")

# What about the 2PN/3PN spin terms in phenomx_tidal_phase?
# These are included in phiTfRef but NOT in psi_T (only NRTidalv3_phase)
# LAL's IMRPhenomX_TidalPhase includes them, and LAL's psi_T does too.
# Let's decompose phiTfRef_taper
m1_v, m2_v, _, _, l1, l2 = float(m1), float(m2), float(chi1), float(chi2), float(lambda1), float(lambda2)
M = m1_v + m2_v
X_A = m1_v / M
X_B = m2_v / M
pfaN = 3.0 / (128.0 * X_A * X_B)

# Compute spin coefficients
from ripplegw.waveforms.NRTidalv3_utils import _get_phenomx_spin_coefficients
c2pn, c3pn, c3p5pn = _get_phenomx_spin_coefficients(theta_intrinsic)
Mf_ref = f_ref_Ms

spin_2pn = pfaN * float(c2pn) / (PI ** (1.0/3.0) * Mf_ref ** (1.0/3.0))
spin_3pn = pfaN * float(c3pn) * (PI * Mf_ref) ** (1.0/3.0)
spin_3p5pn = pfaN * float(c3p5pn) * (PI * Mf_ref) ** (2.0/3.0)

print(f"\nSpin terms in phenomx_tidal_phase at f_ref:")
print(f"  2PN:  {spin_2pn:.10f}")
print(f"  3PN:  {spin_3pn:.10f}")
print(f"  3.5PN: {spin_3p5pn:.10f}")
print(f"  Total spin: {spin_2pn + spin_3pn + spin_3p5pn:.10f}")

# NRTidalv3 rational Pade + PN blend at f_ref
NRphaseNRT_part = phiTfRef_taper - (spin_2pn + spin_3pn + spin_3p5pn)
print(f"  NRphase (NRTidalv3 + PN blend): {NRphaseNRT_part:.10f}")

# Compare with NRTidalv3_phase + QM + SS
print(f"\n  NRTidalv3_phase at f_ref: {NRTidalv3_phase_ref:.10f}")
print(f"  NRTidalv3 + QM + SS: {psi_T_ref_no_taper:.10f}")
print(f"  NRphase - (NRTidalv3 + QM + SS): {NRphaseNRT_part - psi_T_ref_no_taper:.10e}")

# ─── Test 3: Full phase shift comparison ─────────────────────────────────────

print("\n" + "=" * 80)
print("Test 3: Reconstruct the total phase difference from components")
print("=" * 80)

# Generate LAL NRTidalv3 waveform
approximant_nrtidal = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS_NRTidalv3")
laldict = lal.CreateDict()
lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, lambda1)
lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, lambda2)
quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(lambda1)
quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(lambda2)
lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)

hp_nrtidal_lal, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0.0, 0.0, chi1, 0.0, 0.0, chi2,
    dist_SI, 0.0, 0.0, 0, 0, 0,
    df, 20.0, 4096.0, f_ref, laldict, approximant_nrtidal,
)
hp_nrtidal_lal_data = np.array(hp_nrtidal_lal.data.data)
hp_nrtidal_lal_data = hp_nrtidal_lal_data[nonzero[0]:nonzero[-1]+1]

# Frequency grid for ripple
fs = jnp.array(f_lal)

# Generate ripple waveforms
lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(jnp.array([lambda1, lambda2, m1, m2]))
params_nrtidal = jnp.array([Mc, eta, chi1, chi2, lambda_tilde, delta_lambda_tilde, 100.0, 0.0, 0.0, 0.0])
params_xas = jnp.array([Mc, eta, chi1, chi2, 100.0, 0.0, 0.0, 0.0])

from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc

hp_nrtidal_rip, _ = gen_IMRPhenomXAS_NRTidalv3_hphc(fs, params_nrtidal, f_ref)
hp_xas_rip, _ = gen_IMRPhenomXAS_hphc(fs, params_xas, f_ref)

# Phase differences
phase_nrtidal_lal = np.unwrap(np.angle(hp_nrtidal_lal_data))
phase_xas_lal = np.unwrap(np.angle(hp_lal_data))
phase_nrtidal_rip = np.unwrap(np.angle(np.array(hp_nrtidal_rip)))
phase_xas_rip = np.unwrap(np.angle(np.array(hp_xas_rip)))

diff_lal = phase_nrtidal_lal - phase_xas_lal
diff_rip = phase_nrtidal_rip - phase_xas_rip

# Align at f_ref
f_ref_idx = np.argmin(np.abs(f_lal - f_ref))
diff_lal_aligned = diff_lal - diff_lal[f_ref_idx]
diff_rip_aligned = diff_rip - diff_rip[f_ref_idx]

residual = diff_lal_aligned - diff_rip_aligned

# Check: is the residual related to the spin terms?
# If phiTfRef includes spin terms but psi_T doesn't (at frequencies where
# the NRTidalv3 rational Pade is used), there would be a mismatch.
# The residual at frequency f would be: spin_terms(f_ref) - spin_terms(f)
# But spin_terms are frequency-dependent, so this would produce a structured residual.

# Actually, the phase_shift formula is:
# phase_shift = linb * f*M_s + lina + phifRef - 2*pi + 2*pi*f*tc + 2*phic
#
# And phifRef = -(Phase(f_ref) + linb*f_ref*M_s + lina - phiTfRef) + PI/4 + PI
#
# So: phase_shift = linb*(f-f_ref)*M_s - Phase(f_ref) + phiTfRef + PI/4 + PI - 2*pi
#                 = linb*(f-f_ref)*M_s - Phase(f_ref) + phiTfRef - 5*pi/4
#
# Total phase = Phase(f) + phase_shift - psi_T(f) - psi_QM(f) - psi_SS(f)
#
# Phase difference (NRTidalv3 - XAS) at frequency f:
# = phase_shift - psi_T(f) - psi_QM(f) - psi_SS(f)
# = linb*(f-f_ref)*M_s - Phase(f_ref) + phiTfRef - 5*pi/4 - psi_T(f) - psi_QM(f) - psi_SS(f)
#
# At f_ref:
# = -Phase(f_ref) + phiTfRef - 5*pi/4 - psi_T(f_ref) - psi_QM(f_ref) - psi_SS(f_ref)
#
# For this to be 0 (aligned at f_ref):
# phiTfRef = Phase(f_ref) + 5*pi/4 + psi_T(f_ref) + psi_QM(f_ref) + psi_SS(f_ref)
#
# But ripple's phiTfRef = phenomx_tidal_phase(f_ref), which is NOT equal to
# Phase(f_ref) + 5*pi/4 + psi_T(f_ref) + psi_QM(f_ref) + psi_SS(f_ref).
#
# This means the phase difference at f_ref is NOT exactly 0 from the formula alone.
# The alignment happens because both LAL and ripple construct their waveforms
# such that the tidal contribution vanishes at f_ref.
#
# Let me check if the phase difference is truly 0 at f_ref for both.

print(f"\nPhase difference at f_ref:")
print(f"  LAL:   {diff_lal_aligned[f_ref_idx]:.10e}")
print(f"  Ripple: {diff_rip_aligned[f_ref_idx]:.10e}")

# Compute the expected phase difference from the formula
from ripplegw.waveforms.IMRPhenomXAS import Phase as RipplePhaseXAS

Phase_fref = float(RipplePhaseXAS(f_ref, theta_bbh, bbh_phase_coeffs))
print(f"\n  Phase(f_ref) from ripple: {Phase_fref:.6f}")

# linb
delta = jnp.sqrt(1.0 - 4.0 * float(eta))
mm1 = 0.5 * (1.0 + delta)
mm2 = 0.5 * (1.0 - delta)
StotR = (mm1**2 + mm2**2)**(-1.0) * (mm1**2 * chi1 + mm2**2 * chi2)
chia = chi1 - chi2

from ripplegw.waveforms.IMRPhenomX_utils import calc_phaseatpeak, get_cutoff_fMs
lina, linb_init, psi4tostrain = calc_phaseatpeak(float(eta), StotR, chia, delta)
fMs_RD, fMs_damp, _, _ = get_cutoff_fMs(m1, m2, chi1, chi2)
dphi22Ref = jax.grad(RipplePhaseXAS)((fMs_RD - fMs_damp) / M_s, theta_bbh, bbh_phase_coeffs) / M_s
linb_step1 = linb_init - float(dphi22Ref) - 2.0 * PI * (500.0 + float(psi4tostrain))

f_merger = float(_get_merger_frequency(theta_intrinsic))
f_final = min(float(fs[-1]) + df, f_merger)
dphiXAS = float(RipplePhaseDeriv(f_final, theta_bbh, bbh_phase_coeffs) / M_s)
dphiT = float(phenomx_tidal_phase_derivative(theta_intrinsic, f_final * M_s))

dphi_fmerger = dphiXAS + linb_step1 - dphiT
linb_final = linb_step1 - dphi_fmerger

phiTfRef = float(phenomx_tidal_phase(theta_intrinsic, f_ref * M_s))
phifRef = -(Phase_fref + linb_final * f_ref * M_s + lina - phiTfRef) + PI/4.0 + PI

# At an arbitrary frequency, the phase difference should be:
# phase_shift(f) - psi_T(f) - psi_QM(f) - psi_SS(f)
# where phase_shift(f) = linb_final * f*M_s + lina + phifRef - 2*pi

for f_check in [20, 50, 100, 200, 500, 1000]:
    idx = np.argmin(np.abs(f_lal - f_check))
    f_val = float(f_lal[idx])
    fMs = f_val * M_s

    phase_shift = linb_final * fMs + lina + float(phifRef) - 2.0 * PI

    # psi_T at f_val (no taper at low freq)
    x_val = PI * fMs
    NRTidalv3_phase_val = float(get_tidal_phase(x_val, NRTidalv3_coeffs, PN_coeffs))
    psi_QM_val = float(get_qm_phase_correction(fMs, theta_intrinsic))
    psi_SS_val = float(get_spin_phase_correction(x_val ** (2.0/3.0), theta_intrinsic))

    expected_diff = phase_shift - NRTidalv3_phase_val - psi_QM_val - psi_SS_val
    actual_diff = float(diff_rip_aligned[idx])

    print(f"\n  f = {f_val:.2f} Hz:")
    print(f"    phase_shift: {phase_shift:.6f}")
    print(f"    psi_T (NRTidalv3): {NRTidalv3_phase_val:.6f}")
    print(f"    psi_QM: {psi_QM_val:.6f}")
    print(f"    psi_SS: {psi_SS_val:.6f}")
    print(f"    Expected phase diff: {expected_diff:.6f}")
    print(f"    Actual ripple diff:  {actual_diff:.6f}")
    print(f"    Difference: {expected_diff - actual_diff:.6e}")

# ─── Test 4: Is the residual related to spin terms? ──────────────────────────

print("\n" + "=" * 80)
print("Test 4: Residual structure analysis")
print("=" * 80)

# If the residual is caused by spin term mismatch:
# residual(f) = [spin_terms_in_phiTfRef - spin_terms_in_psi](f)
# The spin terms in phiTfRef are computed at f_ref (constant).
# The spin terms in psi_T might be computed differently at each frequency.

# Compute spin terms at each frequency
f_check_arr = np.linspace(20, 2000, 100)
spin_diff_arr = []
residual_arr = []

for f_val in f_check_arr:
    fMs = f_val * M_s
    x_val = PI * fMs

    # Spin terms in phiTfRef (constant, computed at f_ref)
    spin_total_phiTfRef = spin_2pn + spin_3pn + spin_3p5pn

    # Spin terms that would be in psi_T if we used phenomx_tidal_phase at each f
    spin_2pn_f = pfaN * float(c2pn) / (PI ** (1.0/3.0) * (f_val * M_s) ** (1.0/3.0))
    spin_3pn_f = pfaN * float(c3pn) * (PI * f_val * M_s) ** (1.0/3.0)
    spin_3p5pn_f = pfaN * float(c3p5pn) * (PI * f_val * M_s) ** (2.0/3.0)
    spin_total_f = spin_2pn_f + spin_3pn_f + spin_3p5pn_f

    spin_diff_arr.append(spin_total_phiTfRef - spin_total_f)

    idx = np.argmin(np.abs(f_lal - f_val))
    residual_arr.append(float(residual[idx]))

spin_diff_arr = np.array(spin_diff_arr)
residual_arr = np.array(residual_arr)

# Check correlation
from scipy.stats import pearsonr
corr, pval = pearsonr(spin_diff_arr, residual_arr)
print(f"Correlation between spin term difference and residual: {corr:.6f} (p={pval:.2e})")

# Plot would show if the residual tracks the spin term difference
print(f"\n{'f(Hz)':>8} {'Spin diff':>14} {'Residual':>14} {'Ratio':>10}")
for i in [0, 10, 25, 50, 75, 99]:
    ratio = residual_arr[i] / spin_diff_arr[i] if abs(spin_diff_arr[i]) > 1e-15 else float('nan')
    print(f"{f_check_arr[i]:8.1f} {spin_diff_arr[i]:+14.6e} {residual_arr[i]:+14.6e} {ratio:10.4f}")
