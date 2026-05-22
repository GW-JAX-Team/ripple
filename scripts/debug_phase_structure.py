#!/usr/bin/env python
"""Diagnose the remaining ripple vs LAL phase discrepancy for IMRPhenomXAS_NRTidalv3."""

import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.waveforms.IMRPhenomXAS import Phase as RipplePhaseXAS, PhaseDerivative
from ripplegw.waveforms import IMRPhenomX_utils
from ripplegw.waveforms.NRTidalv3_utils import _get_merger_frequency
from ripplegw.constants import MTSUN, PI

# Test parameters
m1, m2 = 1.4, 1.4
chi1, chi2 = 0.0, 0.0
lambda1, lambda2 = 1000.0, 1000.0
f_ref = 20.0
T = 128.0
df = 1.0 / T

Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
M_s = (m1 + m2) * MTSUN
theta_bbh = jnp.array([m1, m2, chi1, chi2])
theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
bbh_phase_coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table

# Generate LAL XAS waveform
m1_kg, m2_kg = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
dist_SI = 100.0 * 1e6 * lal.PC_SI

approx_xas = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")
hp_xas, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0, 0, chi1, 0, 0, chi2,
    dist_SI, 0, 0, 0, 0, 0,
    df, 20.0, 4096.0, f_ref, None, approx_xas,
)

# Generate LAL NRTidalv3 waveform
approx_tidal = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS_NRTidalv3")
laldict = lal.CreateDict()
lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, lambda1)
lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, lambda2)
q1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(lambda1)
q2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(lambda2)
lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, q1 - 1)
lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, q2 - 1)

hp_tidal, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0, 0, chi1, 0, 0, chi2,
    dist_SI, 0, 0, 0, 0, 0,
    df, 20.0, 4096.0, f_ref, laldict, approx_tidal,
)

f_arr = np.arange(len(hp_xas.data.data)) * df
xas_data = np.array(hp_xas.data.data)
tidal_data = np.array(hp_tidal.data.data)

# Find valid (non-zero) region
mask = (np.abs(tidal_data) > 0) & (np.abs(xas_data) > 0)
f_v = f_arr[mask]
xas_v = xas_data[mask]
tidal_v = tidal_data[mask]

print(f"Valid range: {f_v[0]:.2f} to {f_v[-1]:.2f} Hz ({len(f_v)} bins)")

# --- 1. Compare LAL XAS phase with Ripple Phase ---
Phase_rip = np.array([float(RipplePhaseXAS(float(f), theta_bbh, bbh_phase_coeffs)) for f in f_v])
Phase_lal = np.unwrap(np.angle(xas_v))

# Fit a linear model to Phase_rip - Phase_lal
f_Ms_v = f_v * M_s
Mf_v = f_Ms_v
phase_diff_xas = Phase_rip - Phase_lal
# Linear fit: diff = a + b*Mf
A_mat = np.column_stack([np.ones_like(Mf_v), Mf_v])
coeffs = np.linalg.lstsq(A_mat, phase_diff_xas, rcond=None)[0]
linear_fit_xas = coeffs[0] + coeffs[1] * Mf_v
nonlinear_xas = phase_diff_xas - linear_fit_xas
print(f"\nXAS phase difference (ripple - LAL):")
print(f"  Linear component: a={coeffs[0]:.6e}, b={coeffs[1]:.6e}")
print(f"  Nonlinear RMS: {np.std(nonlinear_xas):.6e} rad")

# --- 2. Compare LAL tidal phase with Ripple NRTidal phase ---
# Compute ripple NRTidalv3 waveform
from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3

f_arr_jax = jnp.array(f_v)
params = jnp.array([float(Mc), float(eta), chi1, chi2, lambda1, lambda2, 100.0, 0.0, 0.0])
h_rip = np.array(gen_IMRPhenomXAS_NRTidalv3(f_arr_jax, params, f_ref, use_lambda_tildes=False))

Phase_rip_tidal = np.angle(h_rip)
Phase_rip_tidal_unwrapped = np.unwrap(Phase_rip_tidal)
Phase_lal_tidal = np.unwrap(np.angle(tidal_v))

# Note: LAL also applies tc and phic offsets; ripple uses tc=0, phic=0
# The phase difference will have a linear component (tc, phic offset)
phase_diff_tidal = Phase_rip_tidal_unwrapped - Phase_lal_tidal

# Fit and subtract linear component (tc, phic optimization)
A_mat2 = np.column_stack([np.ones_like(f_v), 2*np.pi*f_v])
coeffs2 = np.linalg.lstsq(A_mat2, phase_diff_tidal, rcond=None)[0]
linear_fit_tidal = coeffs2[0] + coeffs2[1] * 2 * np.pi * f_v
nonlinear_tidal = phase_diff_tidal - linear_fit_tidal
print(f"\nNRTidal v3 phase difference (ripple - LAL), after linear subtraction:")
print(f"  Nonlinear RMS: {np.std(nonlinear_tidal):.6e} rad")
print(f"  Nonlinear max: {np.max(np.abs(nonlinear_tidal)):.6e} rad")

# --- 3. Compare LAL XAS and tidal phases to understand the tidal alignment ---
# From LAL: tidal phase = XAS phase + tidal_correction + linb_change * Mf + phifRef_change
Phase_lal_diff = Phase_lal_tidal - Phase_lal
A_mat3 = np.column_stack([np.ones_like(f_v), Mf_v])
coeffs3 = np.linalg.lstsq(A_mat3, Phase_lal_diff, rcond=None)[0]
linear_fit_lal_diff = coeffs3[0] + coeffs3[1] * Mf_v
nonlinear_lal_diff = Phase_lal_diff - linear_fit_lal_diff
print(f"\nLAL tidal vs XAS phase difference:")
print(f"  Linear: a={coeffs3[0]:.6e}, b={coeffs3[1]:.6e}")
print(f"  Nonlinear RMS: {np.std(nonlinear_lal_diff):.6e} rad")

# --- 4. Inspect dphiXAS values ---
f_merger = float(_get_merger_frequency(theta_intrinsic))
f_final = min(f_v[-1] + df, f_merger)
print(f"\nf_merger = {f_merger:.4f} Hz")
print(f"f_final = {f_final:.4f} Hz, Mf_final = {f_final * M_s:.6f}")

dphiXAS_analytic = float(PhaseDerivative(f_final, theta_bbh, bbh_phase_coeffs) / M_s)
# Secant approximation
dphiXAS_secant = float((RipplePhaseXAS(f_final, theta_bbh, bbh_phase_coeffs)
                         - RipplePhaseXAS(f_final - df, theta_bbh, bbh_phase_coeffs)) / (df * M_s))

print(f"dphiXAS (analytic): {dphiXAS_analytic:.10e}")
print(f"dphiXAS (secant):   {dphiXAS_secant:.10e}")
print(f"diff: {dphiXAS_analytic - dphiXAS_secant:.4e}")

# --- 5. Estimate LAL dphiXAS from the XAS waveform ---
# Extract d(Phase_total)/df from LAL XAS by central differences
dphi_lal_df = np.zeros_like(f_v)
for i in range(2, len(f_v) - 2):
    dphi_lal_df[i] = np.imag(np.conj(xas_v[i]) * (-xas_v[i+2] + 8*xas_v[i+1] - 8*xas_v[i-1] + xas_v[i-2]) / (12 * df)) / np.abs(xas_v[i])**2

# d(Phase_XAS_total)/df at f_final
idx_final = np.argmin(np.abs(f_v - f_final))
dphi_total_lal_df = dphi_lal_df[idx_final]
dphi_total_lal_dMf = dphi_total_lal_df / M_s

# From LAL: d(Phase_total)/dMf = (1/eta) * dPhase22 + linb_BBH
# We don't know linb_BBH directly. But we know from the XAS overlap test:
# Phase_rip(f) ≈ Phase_LAL(f) + linear_offset (a + b*Mf)
# d(Phase_rip)/dMf ≈ d(Phase_LAL)/dMf + b
# So b = d(Phase_rip)/dMf - d(Phase_LAL)/dMf
# From Phase_LAL = inveta*Phase22 + linb*Mf + lina + phifRef:
# d(Phase_LAL)/dMf = inveta*dPhase22 + linb_BBH
# So inveta*dPhase22 = d(Phase_LAL)/dMf - linb_BBH = dphi_total_lal_dMf - linb_BBH
print(f"\nd(Phase_XAS_total)/dMf from LAL at f_final: {dphi_total_lal_dMf:.10e}")
print(f"d(Phase_XAS_rip)/dMf from ripple at f_final: {dphiXAS_analytic:.10e}")
print(f"Difference: {dphiXAS_analytic - dphi_total_lal_dMf:.6e}")
print(f"(This difference = linb_BBH, the BBH time-shift in LAL)")

# The ACTUAL linb_BBH from ripple
from ripplegw.waveforms.IMRPhenomX_utils import calc_phaseatpeak, get_cutoff_fMs
delta = float(jnp.sqrt(1.0 - 4.0 * float(eta)))
mm1 = 0.5 * (1.0 + delta)
mm2 = 0.5 * (1.0 - delta)
StotR = (mm1**2 + mm2**2)**(-1.0) * (mm1**2 * chi1 + mm2**2 * chi2)
chia = chi1 - chi2
lina, linb_raw, psi4tostrain = calc_phaseatpeak(float(eta), StotR, chia, delta)
fMs_RD, fMs_damp, _, _ = get_cutoff_fMs(m1, m2, chi1, chi2)
dphi22Ref = float(jax.grad(RipplePhaseXAS)((fMs_RD - fMs_damp) / M_s, theta_bbh, bbh_phase_coeffs) / M_s)
linb_BBH = float(linb_raw) - dphi22Ref - 2.0 * PI * (500.0 + float(psi4tostrain))
print(f"\nlinb_BBH (ripple): {linb_BBH:.10e}")
print(f"Expected (1/eta)*dPhase22 = dphiXAS_analytic - linb_BBH = {dphiXAS_analytic - linb_BBH:.10e}")
