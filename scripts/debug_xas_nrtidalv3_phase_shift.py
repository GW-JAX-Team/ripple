#!/usr/bin/env python
"""Focused diagnostic: compare LAL vs ripple for IMRPhenomXAS_NRTidalv3 tidal alignment.

Key finding: LAL's dphi_fmerger uses a completely different linb convention.
LAL computes:
    dphi_fmerger = 1/eta * dPhase_22(Mf_final) + linb - dphiT
    tshift = -dphi_fmerger
    linb += tshift

This means: linb_new = linb_old - dphi_fmerger
                          = linb_old - (1/eta * dPhase_22 + linb_old - dphiT)
                          = -1/eta * dPhase_22 + dphiT

Ripple computes:
    linb = linb - (dphiXAS + linb - dphiT)
This is the same formula! So the issue must be in the value of dphiXAS vs 1/eta * dPhase_22.

But wait: at this point in LAL's code, `linb` has already been modified by the
phase-at-peak correction:
    linb = linb_init - dphi22Ref - 2*pi*(500 + psi4tostrain)

So the linb in the tidal correction is the phase-at-peak-corrected linb,
which is what ripple calls `linb_step1`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.conversions import Mc_eta_to_ms, lambdas_to_lambda_tildes, ms_to_Mc_eta
from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import (
    gen_IMRPhenomXAS_NRTidalv3_hphc,
    _gen_IMRPhenomXAS_NRTidalv3,
)
from ripplegw.waveforms.IMRPhenomXAS import (
    gen_IMRPhenomXAS_hphc,
    Phase as RipplePhaseXAS,
    PhaseDerivative as RipplePhaseXASDeriv,
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

m1 = 1.4
m2 = 1.35
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
df = 1.0 / T

Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
    jnp.array([lambda1, lambda2, m1, m2])
)


# ─── LAL waveform generation ─────────────────────────────────────────────────

def get_lal_waveform(m1, m2, chi1, chi2, lambda1, lambda2,
                     dist, tc, phic, inc, f_l, f_u, df, f_ref):
    """Generate LAL waveform."""
    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    dist_SI = dist * 1e6 * lal.PC_SI

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS_NRTidalv3")
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
    return np.array(hp.data.data), np.array(hc.data.data)


def get_lal_waveform_xas(m1, m2, chi1, chi2, dist, tc, phic, inc, f_l, f_u, df, f_ref):
    """Generate plain LAL IMRPhenomXAS waveform."""
    m1_kg = m1 * lal.MSUN_SI
    m2_kg = m2 * lal.MSUN_SI
    dist_SI = dist * 1e6 * lal.PC_SI

    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")

    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1_kg, m2_kg, 0.0, 0.0, chi1, 0.0, 0.0, chi2,
        dist_SI, inc, phic, 0, 0, 0,
        df, f_l, f_u, f_ref, None, approximant,
    )
    return np.array(hp.data.data), np.array(hc.data.data)


print("=" * 80)
print("IMRPhenomXAS_NRTidalv3 LAL vs Ripple Diagnostic (focused)")
print("=" * 80)
print(f"m1={m1}, m2={m2}, chi1={chi1}, chi2={chi2}")
print(f"lambda1={lambda1}, lambda2={lambda2}")
print(f"f_l={f_l}, f_u={f_u}, T={T}, df={df}")

# Generate LAL waveforms
hp_nrtidal_lal, _ = get_lal_waveform(
    m1, m2, chi1, chi2, lambda1, lambda2,
    dist_mpc, tc, phic, inclination, f_l, f_u, df, f_ref
)
hp_xas_lal, _ = get_lal_waveform_xas(
    m1, m2, chi1, chi2,
    dist_mpc, tc, phic, inclination, f_l, f_u, df, f_ref
)

# Find valid frequency range (where both are nonzero)
nonzero_mask = (np.abs(hp_nrtidal_lal) > 0) & (np.abs(hp_xas_lal) > 0)
valid_indices = np.where(nonzero_mask)[0]
f_valid = np.arange(len(hp_nrtidal_lal)) * df
f_valid = f_valid[valid_indices]

hp_nrtidal_lal_v = hp_nrtidal_lal[valid_indices]
hp_xas_lal_v = hp_xas_lal[valid_indices]

print(f"Valid range: {f_valid[0]:.2f} - {f_valid[-1]:.2f} Hz ({len(f_valid)} bins)")

# Extract and unwrap phases
phase_nrtidal_lal = np.unwrap(np.angle(hp_nrtidal_lal_v))
phase_xas_lal = np.unwrap(np.angle(hp_xas_lal_v))
phase_diff_lal = phase_nrtidal_lal - phase_xas_lal

# Align at f_ref
f_ref_idx = np.argmin(np.abs(f_valid - f_ref))
phase_diff_lal_aligned = phase_diff_lal - phase_diff_lal[f_ref_idx]


# ─── Ripple waveforms ────────────────────────────────────────────────────────

fs = jnp.array(f_valid)

params_nrtidal = jnp.array([
    Mc, eta, chi1, chi2, lambda_tilde, delta_lambda_tilde,
    dist_mpc, tc, phic, inclination
])
hp_nrtidal_rip, _ = gen_IMRPhenomXAS_NRTidalv3_hphc(fs, params_nrtidal, f_ref)

params_xas = jnp.array([
    Mc, eta, chi1, chi2,
    dist_mpc, tc, phic, inclination
])
hp_xas_rip, _ = gen_IMRPhenomXAS_hphc(fs, params_xas, f_ref)

phase_nrtidal_rip = np.unwrap(np.angle(np.array(hp_nrtidal_rip)))
phase_xas_rip = np.unwrap(np.angle(np.array(hp_xas_rip)))
phase_diff_rip = phase_nrtidal_rip - phase_xas_rip
phase_diff_rip_aligned = phase_diff_rip - phase_diff_rip[f_ref_idx]

residual = phase_diff_lal_aligned - phase_diff_rip_aligned

print(f"\nPhase difference (NRTidalv3 - XAS) residual (LAL - Ripple):")
print(f"  Mean:   {np.mean(residual):.6e} rad")
print(f"  Std:    {np.std(residual):.6e} rad")
print(f"  Max:    {np.max(np.abs(residual)):.6e} rad")

# Check at specific frequencies
print(f"\n{'f(Hz)':>8} {'LAL':>14} {'Ripple':>14} {'Residual':>14}")
for f_check in [20, 50, 100, 200, 500, 1000, 1500, 2000]:
    idx = np.argmin(np.abs(f_valid - f_check))
    if idx < len(f_valid):
        print(f"{f_valid[idx]:8.2f} {phase_diff_lal_aligned[idx]:+14.6e} "
              f"{phase_diff_rip_aligned[idx]:+14.6e} {residual[idx]:+14.6e}")


# ─── Deep dive: what is the phase shift formula? ─────────────────────────────

print("\n" + "=" * 80)
print("Tidal Alignment Deep Dive")
print("=" * 80)

theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
M_s = (m1 + m2) * MTSUN
bbh_phase_coeffs = PhenomX_phase_coeff_table

delta = jnp.sqrt(1.0 - 4.0 * float(eta))
mm1 = 0.5 * (1.0 + delta)
mm2 = 0.5 * (1.0 - delta)
StotR = (mm1**2 + mm2**2)**(-1.0) * (mm1**2 * chi1 + mm2**2 * chi2)
chia = chi1 - chi2

lina, linb_init, psi4tostrain = calc_phaseatpeak(float(eta), StotR, chia, delta)
fMs_RD, fMs_damp, _, _ = get_cutoff_fMs(m1, m2, chi1, chi2)

dphi22Ref = jax.grad(RipplePhaseXAS)(
    (fMs_RD - fMs_damp) / M_s, jnp.array([m1, m2, chi1, chi2]), bbh_phase_coeffs
) / M_s

linb_step1 = linb_init - float(dphi22Ref) - 2.0 * PI * (500.0 + float(psi4tostrain))

# f_final
f_merger = float(_get_merger_frequency(theta_intrinsic))
f_final = min(float(f_valid[-1]) + df, f_merger)

print(f"f_merger = {f_merger:.4f} Hz")
print(f"f_final = {f_final:.4f} Hz")

# dphiXAS at f_final
dphiXAS = RipplePhaseXASDeriv(f_final, jnp.array([m1, m2, chi1, chi2]), bbh_phase_coeffs) / M_s
print(f"dphiXAS(f_final) = {float(dphiXAS):.10f}")

# dphiT at f_final
dphiT = phenomx_tidal_phase_derivative(theta_intrinsic, f_final * M_s)
print(f"dphiT(f_final) = {float(dphiT):.10f}")

# The key question: does LAL's 1/eta * dPhase_22 match our dphiXAS?
# We can verify by computing d/d f of LAL's XAS phase at f_final
f_final_idx_lal = np.argmin(np.abs(f_valid - f_final))
if f_final_idx_lal > 2:
    # Use higher-order finite difference
    p0 = np.angle(hp_xas_lal_v[f_final_idx_lal - 2])
    p1 = np.angle(hp_xas_lal_v[f_final_idx_lal - 1])
    p2 = np.angle(hp_xas_lal_v[f_final_idx_lal])
    # Unwrap locally
    phases_raw = np.array([p0, p1, p2])
    phases_unwrapped = np.unwrap(phases_raw)
    dphi_lal_fd = (phases_unwrapped[2] - phases_unwrapped[0]) / (2 * df)
    print(f"\nLAL dphiXAS at f_final (3-pt FD): {dphi_lal_fd:.10f}")
    print(f"Ripple dphiXAS at f_final:         {float(dphiXAS):.10f}")
    print(f"Difference:                         {dphi_lal_fd - float(dphiXAS):.10e}")

# Now check the linb tidal correction
# LAL: dphi_fmerger = 1/eta * dPhase_22 + linb - dphiT
# Then: linb_new = linb - dphi_fmerger = -1/eta * dPhase_22 + dphiT
# Note: 1/eta * dPhase_22 = dphiXAS (if Phase already has 1/eta)
# So: linb_new = -dphiXAS + dphiT

# But ripple computes:
# linb = linb - (dphiXAS + linb - dphiT)
#      = -dphiXAS + dphiT

# These are the SAME! So the linb formula is correct.
# The discrepancy must be in dphiXAS itself.

print(f"\n--- linb computation ---")
print(f"linb_step1 (after phase-at-peak): {linb_step1:.10f}")

# Ripple linb tidal correction
dphi_fmerger_rip = float(dphiXAS) + linb_step1 - float(dphiT)
tshift_rip = -dphi_fmerger_rip
linb_final_rip = linb_step1 + tshift_rip
print(f"dphi_fmerger = dphiXAS + linb - dphiT = {dphi_fmerger_rip:.10f}")
print(f"tshift = -dphi_fmerger = {tshift_rip:.10f}")
print(f"linb_final (ripple) = {linb_final_rip:.10f}")

# LAL would compute:
# linb_final_lal = -1/eta * dPhase_22 + dphiT = -dphiXAS + dphiT
linb_final_from_formula = -float(dphiXAS) + float(dphiT)
print(f"linb_final (formula: -dphiXAS + dphiT) = {linb_final_from_formula:.10f}")
print(f"  Difference from ripple: {linb_final_from_formula - linb_final_rip:.10e}")


# ─── What does LAL's phifRef look like? ──────────────────────────────────────

print("\n" + "=" * 80)
print("phifRef reconstruction from LAL waveform")
print("=" * 80)

# The total phase of LAL NRTidalv3 at any frequency is:
# phi_total = Phase_XAS + linb*f*M_s + lina + phifRef - phiTidal - 2*pi + 2*pi*f*tc + 2*phic
# With tc=0, phic=0:
# phi_total = Phase_XAS + linb*f*M_s + lina + phifRef - phiTidal - 2*pi
#
# So: phiTidal = Phase_XAS + linb*f*M_s + lina + phifRef - phi_total - 2*pi
# And: phase_diff = phi_total - Phase_XAS = linb*f*M_s + lina + phifRef - phiTidal - 2*pi

# phiTfRef = -phiTidal(f_ref) [LAL convention]
# At f_ref: phase_diff(f_ref) = linb*f_ref*M_s + lina + phifRef - phiTidal(f_ref) - 2*pi
# But phifRef is chosen so that phase_diff(f_ref) = 0 (aligned at f_ref)
# So: 0 = linb*f_ref*M_s + lina + phifRef + phiTfRef - 2*pi
# => phifRef = -linb*f_ref*M_s - lina - phiTfRef + 2*pi

# Ripple uses: phifRef = -(Phase(f_ref) + linb*f_ref*M_s + lina - phiTfRef) + PI/4 + PI
# But this doesn't match LAL's formula!

# LAL formula (line 748 of LALSimIMRPhenomX.c):
# phifRef = -(inveta * Phase_22(MfRef) + phiTfRef + linb*MfRef + lina) + 2*phi0 + pi/4
#
# Note: phiTfRef in LAL = -IMRPhenomX_TidalPhase(...)  (with minus sign)
# So LAL's phiTfRef is the NEGATIVE of the tidal phase
#
# Ripple's phiTfRef = phenomx_tidal_phase(...)  (positive)
# And ripple uses: phifRef = -(Phase + linb*f_ref*M_s + lina - phiTfRef) + PI/4 + PI
# = -Phase - linb*f_ref*M_s - lina + phiTfRef + PI/4 + PI
#
# LAL: phifRef = -inveta*Phase_22 - phiTfRef_lal - linb*MfRef - lina + 2*phi0 + pi/4
# With phi0 = phic = 0: phifRef = -inveta*Phase_22 - phiTfRef_lal - linb*MfRef - lina + pi/4
#
# Since phiTfRef_lal = -TidalPhase (negative), and ripple phiTfRef = TidalPhase (positive):
# LAL: phifRef = -inveta*Phase_22 + TidalPhase - linb*MfRef - lina + pi/4
# Ripple: phifRef = -Phase - linb*f_ref*M_s - lina + phiTfRef + PI/4 + PI
#
# If Phase = inveta*Phase_22 and phiTfRef = TidalPhase, then:
# LAL: phifRef = -Phase + phiTfRef - linb*f_ref*M_s - lina + pi/4
# Ripple: phifRef = -Phase + phiTfRef - linb*f_ref*M_s - lina + pi/4 + PI
#
# DIFFERENCE: Ripple adds an extra PI!

print(f"\nLAL phifRef formula:")
print(f"  phifRef = -Phase - linb*f*M_s - lina + phiTfRef + pi/4  [assuming phi0=0]")
print(f"Ripple phifRef formula:")
print(f"  phifRef = -Phase - linb*f*M_s - lina + phiTfRef + pi/4 + PI")
print(f"  Extra term in ripple: PI = {PI:.10f}")

# phiTfRef
phiTfRef_rip = float(phenomx_tidal_phase(theta_intrinsic, f_ref * M_s))
print(f"\nphiTfRef (ripple, positive): {phiTfRef_rip:.10f}")
print(f"phiTfRef (LAL convention, negative): {-phiTfRef_rip:.10f}")

# Let's verify by computing the total phase shift at f_ref
# From the LAL waveform:
phase_total_lal_fref = np.unwrap(np.angle(hp_nrtidal_lal_v))[f_ref_idx]
phase_xas_lal_fref = np.unwrap(np.angle(hp_xas_lal_v))[f_ref_idx]

# The total phase at f_ref should be:
# phi_total = Phase_XAS + linb*f*M_s + lina + phifRef - phiTidal - 2*pi
# With our aligned convention (phase_diff = 0 at f_ref):
# phi_total - Phase_XAS = 0 at f_ref

# But LAL doesn't necessarily align at f_ref!
# LAL sets phifRef so that the total phase has a specific form.
# The actual phase at f_ref depends on LAL's tc and phic conventions.

# With tc=0, phic=0, the total phase at f_ref is:
# phi_total(f_ref) = Phase_XAS(f_ref) + linb*f_ref*M_s + lina + phifRef - phiTidal(f_ref) - 2*pi
# And LAL's phifRef = -(Phase_XAS + phiTfRef + linb*f*M_s + lina) + 2*phi0 + pi/4
# = -(Phase_XAS - phiTfRef_rip + linb*f*M_s + lina) + pi/4  [since phiTfRef_lal = -phiTfRef_rip]

# Let me compute both conventions
phase_xas_at_fref = float(RipplePhaseXAS(f_ref, jnp.array([m1, m2, chi1, chi2]), bbh_phase_coeffs))

# LAL phifRef (with phi0=0)
phifRef_lal = -(phase_xas_at_fref - phiTfRef_rip + linb_final_rip * f_ref * M_s + lina) + PI / 4.0
# Ripple phifRef
phifRef_rip = -(phase_xas_at_fref + linb_final_rip * f_ref * M_s + lina - phiTfRef_rip) + PI / 4.0 + PI

print(f"\nphifRef (LAL convention, phi0=0): {phifRef_lal:.10f}")
print(f"phifRef (ripple): {phifRef_rip:.10f}")
print(f"Difference: {phifRef_rip - phifRef_lal:.10f} (= PI = {PI:.10f})")

# Now let's compute the total phase at f_ref for both
# LAL: phi_total = Phase_XAS + linb*f*M_s + lina + phifRef - phiTidal - 2*pi
# But phiTidal(f_ref) = -phiTfRef_lal = phiTfRef_rip
phi_total_lal_at_fref = (
    phase_xas_at_fref
    + linb_final_rip * f_ref * M_s
    + lina
    + phifRef_lal
    - phiTfRef_rip  # phiTidal at f_ref
    - 2.0 * PI
)
print(f"\nTotal phase at f_ref (LAL convention): {phi_total_lal_at_fref:.10f}")
print(f"Total phase at f_ref (LAL waveform):     {phase_total_lal_fref:.10f}")
print(f"XAS phase at f_ref (LAL):                 {phase_xas_lal_fref:.10f}")
print(f"XAS phase at f_ref (ripple):              {phase_xas_at_fref:.10f}")

# The LAL waveform's actual phase at f_ref
# We need to understand LAL's phase convention
# With tc=0, phic=0, the LAL XAS phase at f_ref should be related to phi0
# Let's compute phase_total - phase_xas at f_ref
print(f"\nLAL NRTidalv3 phase at f_ref: {phase_total_lal_fref:.10f}")
print(f"LAL XAS phase at f_ref:       {phase_xas_lal_fref:.10f}")
print(f"Difference:                    {phase_total_lal_fref - phase_xas_lal_fref:.10f}")


# ─── Full waveform overlap ───────────────────────────────────────────────────

print("\n" + "=" * 80)
print("Full Waveform Overlap (with ET_D PSD)")
print("=" * 80)

# Load ET_D PSD
psd_path = "tests/psds/ET_D_psd.txt"
psd_freqs, psd_vals = np.loadtxt(psd_path, unpack=True)

def compute_overlap_loss(h1, h2, psd, freqs):
    """Numerically stable overlap loss computation."""
    # Interpolate PSD
    psd_interp = np.interp(freqs, psd_freqs, psd_vals)
    mask = psd_interp > 0

    h1_m = h1[mask]
    h2_m = h2[mask]
    psd_m = psd_interp[mask]

    inner = np.sum(np.conj(h1_m) * h2_m / psd_m)
    norm1 = np.sqrt(np.sum(np.conj(h1_m) * h1_m / psd_m))
    norm2 = np.sqrt(np.sum(np.conj(h2_m) * h2_m / psd_m))

    overlap = np.abs(inner) / (norm1 * norm2)

    # Stable loss computation
    loss = 2 * (1 - np.sqrt(overlap))
    return float(overlap), float(loss)

overlap_nrtidal, loss_nrtidal = compute_overlap_loss(
    hp_nrtidal_lal_v, np.array(hp_nrtidal_rip), psd_vals, f_valid
)
print(f"Overlap (NRTidalv3): {overlap_nrtidal:.15e}")
print(f"Overlap loss (NRTidalv3): {loss_nrtidal:.15e}")
print(f"log10(overlap loss): {np.log10(loss_nrtidal):.6f}")

# Also check with the aligned phase difference
# If we correct the phase by the residual at each frequency, what's the overlap?
# Create a "corrected" ripple waveform
phase_corrected = np.angle(np.array(hp_nrtidal_rip)) + residual
amp_corrected = np.abs(np.array(hp_nrtidal_rip))
h_corrected = amp_corrected * np.exp(1j * phase_corrected)

overlap_corrected, loss_corrected = compute_overlap_loss(
    hp_nrtidal_lal_v, h_corrected, psd_vals, f_valid
)
print(f"\nAfter phase correction (LAL - Ripple residual removed):")
print(f"Overlap: {overlap_corrected:.15e}")
print(f"Overlap loss: {loss_corrected:.15e}")
print(f"log10(overlap loss): {np.log10(loss_corrected):.6f}")
