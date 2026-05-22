#!/usr/bin/env python
"""Diagnostic script comparing LAL vs ripple intermediate values for IMRPhenomXAS_NRTidalv3.

This script:
1. Generates LAL IMRPhenomXAS_NRTidalv3 and IMRPhenomXAS waveforms
2. Generates ripple IMRPhenomXAS_NRTidalv3 and IMRPhenomXAS waveforms
3. Compares phase differences between tidal and BBH waveforms
4. Diagnoses where the linb/tidal-alignment discrepancy arises
"""

import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.conversions import Mc_eta_to_ms, lambdas_to_lambda_tildes
from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import (
    gen_IMRPhenomXAS_NRTidalv3_hphc,
    _gen_IMRPhenomXAS_NRTidalv3,
    Phase as RipplePhase,
    PhaseDerivative as RipplePhaseDerivative,
)
from ripplegw.waveforms.IMRPhenomXAS import (
    gen_IMRPhenomXAS_hphc,
    Phase as RipplePhaseXAS,
)
from ripplegw.waveforms.IMRPhenomX_utils import (
    calc_phaseatpeak,
    get_cutoff_fMs,
    PhenomX_phase_coeff_table,
)
from ripplegw.waveforms.NRTidalv3_utils import (
    phenomx_tidal_phase,
    phenomx_tidal_phase_derivative,
    fullTidalPhaseCorrection,
    _get_merger_frequency,
)
from ripplegw.constants import MTSUN, PI


# ─── Test parameters ─────────────────────────────────────────────────────────

m1 = 1.4    # solar masses
m2 = 1.35   # solar masses
chi1 = 0.02
chi2 = 0.015
lambda1 = 400.0
lambda2 = 300.0
dist_mpc = 100.0
tc = 0.0
phic = 0.0
inclination = 0.0

f_l = 20.0
f_u = 4096.0
f_ref = 20.0
T = 128.0
f_sampling = 2 * f_u
N = int(T * f_sampling)
df = 1.0 / T


# ─── LAL waveforms ───────────────────────────────────────────────────────────

def get_lal_waveform_internal(waveform, m1, m2, chi1, chi2, lambda1, lambda2,
                               dist, tc, phic, inc, f_l, f_u, df, f_ref):
    """Generate LAL waveform with tidal params."""
    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    dist_SI = dist * 1e6 * lal.PC_SI

    approximant = lalsim.SimInspiralGetApproximantFromString(waveform)

    laldict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, lambda1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, lambda2)
    quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(lambda1)
    quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(lambda2)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)

    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_kg, m2_kg, 0.0, 0.0, chi1, 0.0, 0.0, chi2,
        dist_SI, inc, phic, 0, 0, 0,
        df, f_l, f_u, f_ref, laldict, approximant,
    )

    hp_data = np.array(hp.data.data)
    hc_data = np.array(hc.data.data)

    return hp_data, hc_data


print("=" * 80)
print("IMRPhenomXAS_NRTidalv3 LAL vs Ripple Diagnostic")
print("=" * 80)
print(f"m1={m1}, m2={m2}, chi1={chi1}, chi2={chi2}")
print(f"lambda1={lambda1}, lambda2={lambda2}")
print(f"f_l={f_l}, f_u={f_u}, T={T}, df={df}")
print()

# Generate LAL waveforms
hp_nrtidal_lal, hc_nrtidal_lal = get_lal_waveform_internal(
    "IMRPhenomXAS_NRTidalv3", m1, m2, chi1, chi2, lambda1, lambda2,
    dist_mpc, tc, phic, inclination, f_l, f_u, df, f_ref
)
hp_xas_lal, hc_xas_lal = get_lal_waveform_internal(
    "IMRPhenomXAS", m1, m2, chi1, chi2, 0.0, 0.0,
    dist_mpc, tc, phic, inclination, f_l, f_u, df, f_ref
)

# Trim to non-zero
nonzero_lal = np.where(np.abs(hp_nrtidal_lal) > 0)[0]
if len(nonzero_lal) > 0:
    idx_start = nonzero_lal[0]
    idx_end = nonzero_lal[-1] + 1
    hp_nrtidal_lal = hp_nrtidal_lal[idx_start:idx_end]
    hc_nrtidal_lal = hc_nrtidal_lal[idx_start:idx_end]
    hp_xas_lal = hp_xas_lal[idx_start:idx_end]
    hc_xas_lal = hc_xas_lal[idx_start:idx_end]
    freqs_lal = np.arange(idx_start, idx_end) * df

print(f"LAL: {len(hp_nrtidal_lal)} non-zero frequency bins")

# Extract phases
phase_nrtidal_lal = np.angle(hp_nrtidal_lal)
phase_xas_lal = np.angle(hp_xas_lal)
phase_diff_lal = phase_nrtidal_lal - phase_xas_lal

# Unwrap phase difference
phase_diff_lal_unwrapped = np.unwrap(phase_diff_lal)

# Also compute amplitude ratio for reference
amp_ratio_lal = np.abs(hp_nrtidal_lal) / np.maximum(np.abs(hp_xas_lal), 1e-300)


# ─── Ripple waveforms ────────────────────────────────────────────────────────

# Build frequency grid matching LAL
fs = jnp.array(freqs_lal)

# Ripple NRTidalv3
params_nrtidal = jnp.array([
    # Mc, eta
    *np.array(Mc_eta_to_ms(jnp.array([m1 + m2, m1 * m2 / (m1 + m2)**2]))[0:0]),  # placeholder
])
# Compute Mc and eta properly
from ripplegw.conversions import ms_to_Mc_eta
Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
    jnp.array([lambda1, lambda2, m1, m2])
)
params_nrtidal = jnp.array([
    Mc, eta, chi1, chi2, lambda_tilde, delta_lambda_tilde,
    dist_mpc, tc, phic, inclination
])

hp_nrtidal_rip, hc_nrtidal_rip = gen_IMRPhenomXAS_NRTidalv3_hphc(
    fs, params_nrtidal, f_ref
)

# Ripple XAS (BBH baseline)
params_xas = jnp.array([
    Mc, eta, chi1, chi2,
    dist_mpc, tc, phic, inclination
])
hp_xas_rip, hc_xas_rip = gen_IMRPhenomXAS_hphc(fs, params_xas, f_ref)

print(f"Ripple: {len(hp_nrtidal_rip)} frequency bins")

# Extract phases
phase_nrtidal_rip = np.angle(np.array(hp_nrtidal_rip))
phase_xas_rip = np.angle(np.array(hp_xas_rip))
phase_diff_rip = phase_nrtidal_rip - phase_xas_rip
phase_diff_rip_unwrapped = np.unwrap(phase_diff_rip)


# ─── Compare phase differences ──────────────────────────────────────────────

print("\n" + "=" * 80)
print("Phase Difference Comparison (NRTidalv3 - XAS)")
print("=" * 80)

# Unwrap both from the same starting point
# The absolute phase offset doesn't matter; we care about the shape
# Subtract the value at f_ref to align
f_ref_idx_lal = np.argmin(np.abs(freqs_lal - f_ref))
f_ref_idx_rip = f_ref_idx_lal  # same grid

phase_diff_lal_aligned = phase_diff_lal_unwrapped - phase_diff_lal_unwrapped[f_ref_idx_lal]
phase_diff_rip_aligned = phase_diff_rip_unwrapped - phase_diff_rip_unwrapped[f_ref_idx_rip]

phase_diff_residual = phase_diff_lal_aligned - phase_diff_rip_aligned

print(f"\nAt f_ref = {f_ref} Hz:")
print(f"  LAL   phase_diff (aligned): {phase_diff_lal_aligned[f_ref_idx_lal]:.6e} rad")
print(f"  Ripple phase_diff (aligned): {phase_diff_rip_aligned[f_ref_idx_rip]:.6e} rad")

print(f"\nResidual (LAL - Ripple) statistics:")
print(f"  Mean:   {np.mean(phase_diff_residual):.6e} rad")
print(f"  Std:    {np.std(phase_diff_residual):.6e} rad")
print(f"  Max:    {np.max(np.abs(phase_diff_residual)):.6e} rad")

# Check at specific frequencies
for f_check in [20, 50, 100, 200, 500, 1000, 2000, 4000]:
    idx = np.argmin(np.abs(freqs_lal - f_check))
    if idx < len(freqs_lal):
        print(f"  f={f_check:5.0f} Hz: LAL={phase_diff_lal_aligned[idx]:+.6e}, "
              f"Rip={phase_diff_rip_aligned[idx]:+.6e}, "
              f"res={phase_diff_residual[idx]:+.6e} rad")


# ─── Compare linb computation ────────────────────────────────────────────────

print("\n" + "=" * 80)
print("linb / Tidal Alignment Computation")
print("=" * 80)

theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
M_s = (m1 + m2) * MTSUN
bbh_phase_coeffs = PhenomX_phase_coeff_table

delta = jnp.sqrt(1.0 - 4.0 * eta)
mm1 = 0.5 * (1.0 + delta)
mm2 = 0.5 * (1.0 - delta)
StotR = (mm1**2 + mm2**2)**(-1.0) * (mm1**2 * chi1 + mm2**2 * chi2)
chia = chi1 - chi2

# calc_phaseatpeak
lina, linb_init, psi4tostrain = calc_phaseatpeak(eta, StotR, chia, delta)
print(f"\nInitial linb (from calc_phaseatpeak): {linb_init:.10f}")
print(f"psi4tostrain: {psi4tostrain:.10f}")

# dphi22Ref
fMs_RD, fMs_damp, _, _ = get_cutoff_fMs(m1, m2, chi1, chi2)
dphi22Ref = jax.grad(RipplePhaseXAS)(
    (fMs_RD - fMs_damp) / M_s, jnp.array([m1, m2, chi1, chi2]), bbh_phase_coeffs
) / M_s
print(f"fMs_RD: {fMs_RD:.6f}, fMs_damp: {fMs_damp:.6f}")
print(f"dphi22Ref: {dphi22Ref:.10f}")

linb_step1 = linb_init - dphi22Ref - 2.0 * PI * (500.0 + psi4tostrain)
print(f"linb after dphi22Ref correction: {linb_step1:.10f}")

# phiTfRef
phiTfRef_rip = phenomx_tidal_phase(theta_intrinsic, f_ref * M_s)
print(f"phiTfRef (ripple): {phiTfRef_rip:.10f}")

# dphiXAS and dphiT at f_final
f_merger = _get_merger_frequency(theta_intrinsic)
f_final_raw = fs[-1] + df
f_final = min(float(f_final_raw), float(f_merger))
print(f"\nf[-1]: {float(fs[-1]):.6f}")
print(f"df: {df:.6f}")
print(f"f[-1]+df: {float(f_final_raw):.6f}")
print(f"f_merger: {float(f_merger):.6f}")
print(f"f_final (capped): {f_final:.6f}")

# dphiXAS - finite difference
dphiXAS_fd = (
    RipplePhaseXAS(f_final, jnp.array([m1, m2, chi1, chi2]), bbh_phase_coeffs)
    - RipplePhaseXAS(f_final - df, jnp.array([m1, m2, chi1, chi2]), bbh_phase_coeffs)
) / (df * M_s)
print(f"dphiXAS (finite diff): {float(dphiXAS_fd):.10f}")

# dphiXAS - analytical
dphiXAS_analytic = RipplePhaseDerivative(f_final, jnp.array([m1, m2, chi1, chi2]), bbh_phase_coeffs) / M_s
print(f"dphiXAS (analytic):  {float(dphiXAS_analytic):.10f}")

# dphiT
dphiT = phenomx_tidal_phase_derivative(theta_intrinsic, f_final * M_s)
print(f"dphiT: {float(dphiT):.10f}")

# linb tidal correction (LAL formula: linb += -(dphi_fmerger) where dphi_fmerger = dphiXAS + linb - dphiT)
dphi_fmerger_rip = float(dphiXAS_analytic) + float(linb_step1) - float(dphiT)
tshift_rip = -dphi_fmerger_rip
linb_final_rip = float(linb_step1) + tshift_rip
print(f"\ndphi_fmerger (ripple): {dphi_fmerger_rip:.10f}")
print(f"tshift (ripple): {tshift_rip:.10f}")
print(f"linb_final (ripple): {linb_final_rip:.10f}")

# What would LAL's 1/eta * dPhase_22 give?
# LAL formula: dphi_fmerger = 1/eta * IMRPhenomX_dPhase_22 + linb - dphiT
# The 1/eta * dPhase_22 should equal our dphiXAS if Phase already has 1/eta
dphi_XAS_scaled = float(dphiXAS_analytic)  # already has 1/eta
print(f"\nVerification: dphiXAS already includes 1/eta factor: {dphi_XAS_scaled:.10f}")
print(f"  1/eta = {1/float(eta):.10f}")


# ─── Compute LAL's dphiXAS from waveform ─────────────────────────────────────
# Extract the phase from LAL's IMRPhenomXAS waveform and compute derivative
# at f_final to compare with ripple

print("\n" + "=" * 80)
print("LAL Phase Derivative at f_final (from waveform)")
print("=" * 80)

# Find f_final index in LAL grid
f_final_idx_lal = np.argmin(np.abs(freqs_lal - f_final))
if f_final_idx_lal > 0 and f_final_idx_lal < len(freqs_lal) - 1:
    # Finite difference from LAL XAS phase
    phase_xas_lal_at_f = np.angle(hp_xas_lal[f_final_idx_lal])
    phase_xas_lal_at_f_minus_df = np.angle(hp_xas_lal[f_final_idx_lal - 1])
    # Unwrap locally
    dphase_lal = np.unwrap([phase_xas_lal_at_f_minus_df, phase_xas_lal_at_f])[1] - \
                 np.unwrap([phase_xas_lal_at_f_minus_df, phase_xas_lal_at_f])[0]
    dphiXAS_lal_fd = dphase_lal / df
    print(f"LAL dphiXAS (finite diff at f_final): {dphiXAS_lal_fd:.10f}")
    print(f"Ripple dphiXAS (finite diff at f_final): {float(dphiXAS_fd):.10f}")
    print(f"Difference: {dphiXAS_lal_fd - float(dphiXAS_fd):.10e}")


# ─── Compare with lambda=0 ──────────────────────────────────────────────────

print("\n" + "=" * 80)
print("Zero-lambda comparison (NRTidalv3 with lambda=0 vs plain XAS)")
print("=" * 80)

# LAL with lambda=0
hp_nrtidal_lal_l0, _ = get_lal_waveform_internal(
    "IMRPhenomXAS_NRTidalv3", m1, m2, chi1, chi2,
    1e-10, 1e-10,  # tiny nonzero to avoid LAL issues
    dist_mpc, tc, phic, inclination, f_l, f_u, df, f_ref
)

# Ripple with lambda=0
params_l0 = jnp.array([
    Mc, eta, chi1, chi2, 0.0, 0.0,  # lambda_tilde=0, delta_lambda=0
    dist_mpc, tc, phic, inclination
])
hp_nrtidal_rip_l0, _ = gen_IMRPhenomXAS_NRTidalv3_hphc(fs, params_l0, f_ref)

# Compare zero-lambda NRTidalv3 to plain XAS
phase_nrtidal_lal_l0 = np.angle(hp_nrtidal_lal_l0[idx_start:idx_end])
phase_nrtidal_rip_l0 = np.angle(np.array(hp_nrtidal_rip_l0))

# Phase difference: NRTidalv3(lambda=0) - XAS
phase_diff_lal_l0 = np.unwrap(phase_nrtidal_lal_l0) - np.unwrap(phase_xas_lal)
phase_diff_rip_l0 = np.unwrap(phase_nrtidal_rip_l0) - np.unwrap(np.array(hp_xas_rip))

# Align at f_ref
phase_diff_lal_l0_aligned = phase_diff_lal_l0 - phase_diff_lal_l0[f_ref_idx_lal]
phase_diff_rip_l0_aligned = phase_diff_rip_l0 - phase_diff_rip_l0[f_ref_idx_rip]
residual_l0 = phase_diff_lal_l0_aligned - phase_diff_rip_l0_aligned

print(f"\nZero-lambda residual (LAL - Ripple) statistics:")
print(f"  Mean:   {np.mean(residual_l0):.6e} rad")
print(f"  Std:    {np.std(residual_l0):.6e} rad")
print(f"  Max:    {np.max(np.abs(residual_l0)):.6e} rad")

for f_check in [20, 50, 100, 200, 500, 1000, 2000, 4000]:
    idx = np.argmin(np.abs(freqs_lal - f_check))
    if idx < len(freqs_lal):
        print(f"  f={f_check:5.0f} Hz: LAL={phase_diff_lal_l0_aligned[idx]:+.6e}, "
              f"Rip={phase_diff_rip_l0_aligned[idx]:+.6e}, "
              f"res={residual_l0[idx]:+.6e} rad")


# ─── Full waveform overlap ───────────────────────────────────────────────────

print("\n" + "=" * 80)
print("Full Waveform Overlap")
print("=" * 80)

# Simple inner product
def simple_inner_product(h1, h2, freqs):
    """Noise-weighted inner product (flat PSD for simplicity)."""
    # Only use frequencies above f_l
    mask = freqs >= f_l
    h1 = h1[mask]
    h2 = h2[mask]
    freqs_masked = freqs[mask]

    # Simple overlap (no PSD weighting)
    inner = np.sum(np.conj(h1) * h2 / freqs_masked).real
    norm1 = np.sqrt(np.sum(np.conj(h1) * h1 / freqs_masked).real)
    norm2 = np.sqrt(np.sum(np.conj(h2) * h2 / freqs_masked).real)
    return inner / (norm1 * norm2)

overlap = simple_inner_product(hp_nrtidal_lal, np.array(hp_nrtidal_rip), freqs_lal)
overlap_loss = 1 - overlap
print(f"Overlap: {overlap:.15e}")
print(f"Overlap loss: {overlap_loss:.15e}")
print(f"log10(overlap loss): {np.log10(overlap_loss):.6f}")


# ─── phifRef comparison ──────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("phifRef computation")
print("=" * 80)

phifRef_rip = -(
    RipplePhaseXAS(f_ref, jnp.array([m1, m2, chi1, chi2]), bbh_phase_coeffs)
    + linb_final_rip * (f_ref * M_s)
    + lina
    - phiTfRef_rip
) + PI / 4.0 + PI

print(f"phifRef (ripple): {float(phifRef_rip):.10f}")

# Try to infer LAL's phifRef from the waveform
# The total phase of LAL NRTidalv3 at f_ref should be:
# phase_total(f_ref) = Phase_XAS(f_ref) + linb*f_ref*M_s + lina + phifRef - phiT(f_ref) - 2*pi
# So: phifRef = phase_total(f_ref) - Phase_XAS(f_ref) - linb*f_ref*M_s - lina + phiT(f_ref) + 2*pi

# Get LAL NRTidalv3 phase at f_ref (unwrapped from a consistent reference)
# Use the raw complex value at f_ref
h_nrtidal_lal_fref = hp_nrtidal_lal[f_ref_idx_lal]
h_xas_lal_fref = hp_xas_lal[f_ref_idx_lal]

phase_total_lal = np.angle(h_nrtidal_lal_fref)
phase_xas_lal_at_fref = np.angle(h_xas_lal_fref)

# The LAL convention for phifRef can be inferred
# We need to figure out LAL's phi0 convention
# In LAL: phifRef = -(1/eta * Phase_22(MfRef) + phiTfRef + linb*MfRef + lina) + 2*phi0 + pi/4
# The total phase at f_ref is: 1/eta*Phase_22 + phiTfRef + linb*MfRef + lina + phifRef + tc/freq stuff
# = 1/eta*Phase_22 + phiTfRef + linb*MfRef + lina
#   -(1/eta * Phase_22(MfRef) + phiTfRef + linb*MfRef + lina) + 2*phi0 + pi/4
# = 2*phi0 + pi/4

# With tc=0, phic=0, phi0 = phic (approximately for LAL)
# So total phase at f_ref = 2*0 + pi/4 = pi/4
print(f"\nLAL total phase at f_ref (raw): {phase_total_lal:.10f}")
print(f"LAL XAS phase at f_ref (raw):   {phase_xas_lal_at_fref:.10f}")
print(f"Expected from phifRef formula:  {float(PI / 4.0):.10f} (if phi0=0)")

# The difference should tell us about the phase convention
phase_at_fref_diff = phase_total_lal - phase_xas_lal_at_fref
print(f"Difference: {phase_at_fref_diff:.10f}")
print(f"pi/4 = {float(PI/4):.10f}")
