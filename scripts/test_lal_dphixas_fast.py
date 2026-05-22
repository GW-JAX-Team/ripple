#!/usr/bin/env python
"""Fast test: use LAL's dphiXAS in the ripple linb formula and check overlap."""

import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.waveforms.IMRPhenomXAS import Phase as RipplePhaseXAS
from ripplegw.waveforms.IMRPhenomX_utils import (
    PhenomX_phase_coeff_table,
    calc_phaseatpeak,
    get_cutoff_fMs,
)
from ripplegw.waveforms.NRTidalv3_utils import (
    phenomx_tidal_phase,
    phenomx_tidal_phase_derivative,
    _get_merger_frequency,
    general_planck_taper,
)
from ripplegw.constants import MTSUN, PI

m1, m2 = 1.4, 1.35
chi1, chi2 = 0.02, 0.015
lambda1, lambda2 = 400.0, 300.0
f_ref = 20.0
T = 128.0
df_val = 1.0 / T

Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(jnp.array([lambda1, lambda2, m1, m2]))
M_s = (m1 + m2) * MTSUN
theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
theta_bbh = jnp.array([m1, m2, chi1, chi2])
bbh_phase_coeffs = PhenomX_phase_coeff_table

# Generate LAL waveforms
m1_kg = m1 * lal.MSUN_SI
m2_kg = m2 * lal.MSUN_SI
dist_SI = 100.0 * 1e6 * lal.PC_SI

approx_xas = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")
hp_xas_lal, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0.0, 0.0, chi1, 0.0, 0.0, chi2,
    dist_SI, 0.0, 0.0, 0, 0, 0,
    df_val, 20.0, 4096.0, f_ref, None, approx_xas,
)

approx_nrtidal = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS_NRTidalv3")
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
    df_val, 20.0, 4096.0, f_ref, laldict, approx_nrtidal,
)

hp_xas_data = np.array(hp_xas_lal.data.data)
hp_nrtidal_data = np.array(hp_nrtidal_lal.data.data)
f_arr = np.arange(len(hp_xas_data)) * df_val

nonzero = np.where((np.abs(hp_nrtidal_data) > 0) & (np.abs(hp_xas_data) > 0))[0]
f_arr_v = f_arr[nonzero]
hp_nrtidal_v = hp_nrtidal_data[nonzero]
hp_xas_v = hp_xas_data[nonzero]

print(f"Valid range: {f_arr_v[0]:.2f} - {f_arr_v[-1]:.2f} Hz")

# Extract LAL's d(Phase_XAS)/df
h_conj = np.conj(hp_xas_v)
dh_df = np.zeros_like(hp_xas_v, dtype=complex)
for i in range(2, len(hp_xas_v) - 2):
    dh_df[i] = (-hp_xas_v[i+2] + 8*hp_xas_v[i+1] - 8*hp_xas_v[i-1] + hp_xas_v[i-2]) / (12 * df_val)

dphi_xas_df = np.imag(h_conj * dh_df) / (np.abs(hp_xas_v) ** 2)
# Replace NaN at edges
dphi_xas_df[:2] = dphi_xas_df[2]
dphi_xas_df[-2:] = dphi_xas_df[-3]

# LAL dphiXAS = d(Phase_XAS)/dMf = d(Phase_XAS)/df / M_s * M_s... no
# d(Phase)/dMf = d(Phase)/df / (d(f*M_s)/df) = d(Phase)/df / M_s
dphi_xas_dMf = dphi_xas_df / M_s

# Key comparison
f_merger = float(_get_merger_frequency(theta_intrinsic))
f_final = min(float(f_arr_v[-1]) + df_val, f_merger)
f_final_idx = np.argmin(np.abs(f_arr_v - f_final))

print(f"\nf_final = {f_final:.4f} Hz (index {f_final_idx})")
print(f"d(Phase_XAS)/df from LAL: {dphi_xas_df[f_final_idx]:+.10e}")
print(f"dphiXAS (LAL, d/dMf): {dphi_xas_dMf[f_final_idx]:+.10e}")

from ripplegw.waveforms.IMRPhenomXAS import PhaseDerivative as RipplePhaseDeriv
dphiXAS_rip = float(RipplePhaseDeriv(f_final, theta_bbh, bbh_phase_coeffs) / M_s)
print(f"dphiXAS (ripple, d/dMf): {dphiXAS_rip:+.10e}")

# linb computation
delta = jnp.sqrt(1.0 - 4.0 * float(eta))
mm1 = 0.5 * (1.0 + delta)
mm2 = 0.5 * (1.0 - delta)
StotR = (mm1**2 + mm2**2)**(-1.0) * (mm1**2 * chi1 + mm2**2 * chi2)
chia = chi1 - chi2

lina, linb_init, psi4tostrain = calc_phaseatpeak(float(eta), StotR, chia, delta)
fMs_RD, fMs_damp, _, _ = get_cutoff_fMs(m1, m2, chi1, chi2)
dphi22Ref = jax.grad(RipplePhaseXAS)((fMs_RD - fMs_damp) / M_s, theta_bbh, bbh_phase_coeffs) / M_s
linb_step1 = linb_init - float(dphi22Ref) - 2.0 * PI * (500.0 + float(psi4tostrain))

dphiT = float(phenomx_tidal_phase_derivative(theta_intrinsic, f_final * M_s))

# With LAL dphiXAS
dphi_fmerger_lal = dphi_xas_dMf[f_final_idx] + linb_step1 - dphiT
linb_final_lal = linb_step1 - dphi_fmerger_lal

# With ripple dphiXAS
dphi_fmerger_rip = dphiXAS_rip + linb_step1 - dphiT
linb_final_rip = linb_step1 - dphi_fmerger_rip

print(f"\nlinb_final (LAL dphiXAS): {linb_final_lal:.10f}")
print(f"linb_final (ripple dphiXAS): {linb_final_rip:.10f}")
print(f"Difference: {linb_final_lal - linb_final_rip:.10f}")

# Now compute the NRTidalv3 phase with both linb values and compare overlap
# Use vectorized operations
fs = jnp.array(f_arr_v)
f_Ms = fs * M_s

# Vectorized Phase computation
@jax.jit
def batch_phase(f_arr):
    def single(f):
        return RipplePhaseXAS(f, theta_bbh, bbh_phase_coeffs)
    return jax.lax.map(single, f_arr)

Phase_arr = np.array(batch_phase(fs))
Phase_fref = float(RipplePhaseXAS(f_ref, theta_bbh, bbh_phase_coeffs))

# phiTfRef
phiTfRef = float(phenomx_tidal_phase(theta_intrinsic, f_ref * M_s))

# phifRef for both linb values
phifRef_lal = -(Phase_fref + float(linb_final_lal) * f_ref * M_s + lina - phiTfRef) + PI/4.0 + PI
phifRef_rip = -(Phase_fref + float(linb_final_rip) * f_ref * M_s + lina - phiTfRef) + PI/4.0 + PI

# Phase shifts
phase_shift_lal = float(linb_final_lal) * f_Ms + lina + phifRef_lal - 2.0 * PI
phase_shift_rip = float(linb_final_rip) * f_Ms + lina + phifRef_rip - 2.0 * PI

# Tidal phase (vectorized)
@jax.jit
def batch_tidal_phase(f_arr):
    from ripplegw.waveforms.NRTidalv3_utils import (
        get_tidal_phase, get_tidal_phase_PN, get_tidalphasePN_coeffs,
        get_NRTidalv3_coefficients, general_planck_taper,
    )
    from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import (
        get_qm_phase_correction, get_spin_phase_correction, get_planck_taper,
        get_tidal_amplitude, get_kappa,
    )

    PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
    NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)
    f_merger_val = float(_get_merger_frequency(theta_intrinsic))

    def single(f):
        fMs = f * M_s
        x = PI * fMs
        P_P = general_planck_taper(fMs, 1.15 * f_merger_val * M_s, 1.35 * f_merger_val * M_s)
        NRTidalv3_phase = get_tidal_phase(x, NRTidalv3_coeffs, PN_coeffs)
        PN_tidal = get_tidal_phase_PN(x, float(m1)/(m1+m2), float(lambda1), float(lambda2), PN_coeffs)
        psi_T = NRTidalv3_phase * (1 - P_P) + PN_tidal * P_P
        psi_QM = get_qm_phase_correction(fMs, theta_intrinsic)
        psi_SS = get_spin_phase_correction(x ** (2.0/3.0), theta_intrinsic)
        return psi_T + psi_QM + psi_SS

    return jax.lax.map(single, f_arr)

tidal_total = np.array(batch_tidal_phase(fs))

# Total phase
total_phase_lal = Phase_arr + phase_shift_lal - tidal_total
total_phase_rip = Phase_arr + phase_shift_rip - tidal_total

# Amplitude
bbh_amp = np.abs(hp_xas_v)

@jax.jit
def batch_amp_correction(f_arr):
    from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import (
        get_planck_taper, get_tidal_amplitude, get_kappa,
    )
    f_merger_val = float(_get_merger_frequency(theta_intrinsic))

    def single(f):
        fMs = f * M_s
        x = PI * fMs
        x_23 = x ** (2.0/3.0)
        kappa = get_kappa(theta=theta_intrinsic)
        A_T = get_tidal_amplitude(x_23, theta_intrinsic, kappa, distance=100.0)
        A_P = get_planck_taper(jnp.array(f), f_merger_val)
        return A_P * (1.0 + A_T / bbh_amp[np.argmin(np.abs(f_arr - float(f)))])

    return jax.lax.map(single, f_arr)

# For simplicity, use LAL's amplitude directly
strain_lal = hp_xas_v * np.exp(1j * (total_phase_lal - np.angle(hp_xas_v)))
strain_rip = hp_xas_v * np.exp(1j * (total_phase_rip - np.angle(hp_xas_v)))

# Overlap loss
def overlap(h1, h2):
    mask = np.abs(h1) > 0
    h1, h2 = h1[mask], h2[mask]
    psd = 1.0 / np.arange(1, len(h1)+1)  # Simple weighting
    inner = np.sum(np.conj(h1) * h2)
    norm1 = np.sqrt(np.sum(np.conj(h1) * h1))
    norm2 = np.sqrt(np.sum(np.conj(h2) * h2))
    return np.abs(inner) / (norm1 * norm2)

ov_lal = overlap(hp_nrtidal_v, strain_lal)
ov_rip = overlap(hp_nrtidal_v, strain_rip)
loss_lal = 1 - ov_lal
loss_rip = 1 - ov_rip

print(f"\nOverlap (LAL dphiXAS): {ov_lal:.15e}")
print(f"Overlap loss (LAL dphiXAS): {loss_lal:.15e}")
print(f"log10(loss): {np.log10(loss_lal):.4f}")

print(f"\nOverlap (ripple dphiXAS): {ov_rip:.15e}")
print(f"Overlap loss (ripple dphiXAS): {loss_rip:.15e}")
print(f"log10(loss): {np.log10(loss_rip):.4f}")
