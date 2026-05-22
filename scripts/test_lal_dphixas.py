#!/usr/bin/env python
"""Test: use numerical dphiXAS extracted from LAL XAS waveform directly."""

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
    get_tidal_phase,
    get_tidal_phase_PN,
    get_tidalphasePN_coeffs,
    get_NRTidalv3_coefficients,
)
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import (
    get_planck_taper,
    get_tidal_amplitude,
    get_kappa,
    get_qm_phase_correction,
    get_spin_phase_correction,
)
from ripplegw.constants import MTSUN, PI

# Test parameters
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

# Generate LAL XAS waveform
m1_kg = m1 * lal.MSUN_SI
m2_kg = m2 * lal.MSUN_SI
dist_SI = 100.0 * 1e6 * lal.PC_SI

approx_xas = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")
hp_xas_lal, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0.0, 0.0, chi1, 0.0, 0.0, chi2,
    dist_SI, 0.0, 0.0, 0, 0, 0,
    df_val, 20.0, 4096.0, f_ref, None, approx_xas,
)
hp_xas_lal_data = np.array(hp_xas_lal.data.data)
f_arr = np.arange(len(hp_xas_lal_data)) * df_val

# Compute d(Phase_XAS)/df from LAL complex strain
h_conj = np.conj(hp_xas_lal_data)
dh_df = np.zeros_like(hp_xas_lal_data, dtype=complex)
for i in range(2, len(hp_xas_lal_data) - 2):
    dh_df[i] = (-hp_xas_lal_data[i+2] + 8*hp_xas_lal_data[i+1] - 8*hp_xas_lal_data[i-1] + hp_xas_lal_data[i-2]) / (12 * df_val)

dphi_xas_lal_df = np.imag(h_conj * dh_df) / (np.abs(hp_xas_lal_data) ** 2)

# Generate LAL NRTidalv3
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
hp_nrtidal_lal_data = np.array(hp_nrtidal_lal.data.data)

# Non-zero region
nonzero = np.where((np.abs(hp_nrtidal_lal_data) > 0) & (np.abs(hp_xas_lal_data) > 0))[0]
f_arr_v = f_arr[nonzero]
dphi_xas_lal_df_v = dphi_xas_lal_df[nonzero]
hp_nrtidal_lal_v = hp_nrtidal_lal_data[nonzero]
hp_xas_lal_v = hp_xas_lal_data[nonzero]

print(f"Valid range: {f_arr_v[0]:.2f} - {f_arr_v[-1]:.2f} Hz ({len(f_arr_v)} bins)")

# Build custom ripple NRTidalv3 using LAL's d(Phase_XAS)/df
fs = jnp.array(f_arr_v)

# Get LAL's d(Phase_XAS)/dMf at each frequency (this is what dphiXAS should be)
# dphiXAS = d(Phase_XAS)/dMf = d(Phase_XAS)/df / M_s * M_s^2 ... no wait
# d(Phase)/dMf = d(Phase)/df / (dMf/df) = d(Phase)/df / M_s

# Actually, let me think about this differently.
# LAL computes dphi_fmerger = 1/eta * dPhase_22/dMf + linb - dphiT
# where dPhase_22/dMf is the raw 22-mode phase derivative wrt Mf.
#
# In ripple, Phase = (1/eta) * phi_22 + ..., so d(Phase)/dMf = (1/eta) * dphi_22/dMf.
# So LAL's 1/eta * dPhase_22/dMf = d(Phase)/dMf.
#
# From the LAL complex strain: d(Phase_total)/df = dphi_xas_lal_df_v
# d(Phase_total)/dMf = d(Phase_total)/df / M_s (since dMf/df = M_s)
#
# So the correct dphiXAS for LAL should be:
# dphiXAS_lal = dphi_xas_lal_df_v / M_s

# But wait, this is d(Phase_total)/dMf for the XAS waveform.
# The XAS total phase is: Phase_XAS_total = (1/eta) * phi_22 + phifRef_XAS
# d(Phase_XAS_total)/dMf = (1/eta) * dphi_22/dMf = d(Phase_XAS)/dMf (where Phase_XAS = ripple's Phase)

# So dphiXAS_lal = dphi_xas_lal_df_v / M_s should equal ripple's PhaseDerivative / M_s.
# But earlier diagnostics showed they have opposite signs!

# Let me verify:
f_final_idx = np.argmin(np.abs(f_arr_v - 2037.5))
dphi_xas_lal_at_f = dphi_xas_lal_df_v[f_final_idx]
print(f"\nd(Phase_XAS)/df from LAL at f_final: {dphi_xas_lal_at_f:+.10e}")
print(f"d(Phase_XAS)/dMf from LAL: {dphi_xas_lal_at_f / M_s:+.10e}")

# Compare with ripple's PhaseDerivative / M_s
from ripplegw.waveforms.IMRPhenomXAS import PhaseDerivative as RipplePhaseDeriv
f_final = f_arr_v[f_final_idx]
dphi_xas_rip = float(RipplePhaseDeriv(f_final, theta_bbh, bbh_phase_coeffs) / M_s)
print(f"PhaseDerivative / M_s from ripple: {dphi_xas_rip:+.10e}")

# These should be equal if both compute d(Phase)/dMf
# But they're probably opposite signs due to convention differences

# Let me try using LAL's value directly in the ripple linb computation
# and see if the overlap improves.

# Compute ripple's NRTidalv3 with LAL's dphiXAS
delta = jnp.sqrt(1.0 - 4.0 * float(eta))
mm1 = 0.5 * (1.0 + delta)
mm2 = 0.5 * (1.0 - delta)
StotR = (mm1**2 + mm2**2)**(-1.0) * (mm1**2 * chi1 + mm2**2 * chi2)
chia = chi1 - chi2

lina, linb_init, psi4tostrain = calc_phaseatpeak(float(eta), StotR, chia, delta)
fMs_RD, fMs_damp, _, _ = get_cutoff_fMs(m1, m2, chi1, chi2)
dphi22Ref = jax.grad(RipplePhaseXAS)((fMs_RD - fMs_damp) / M_s, theta_bbh, bbh_phase_coeffs) / M_s
linb_step1 = linb_init - float(dphi22Ref) - 2.0 * PI * (500.0 + float(psi4tostrain))

f_merger = float(_get_merger_frequency(theta_intrinsic))
f_final_hz = min(float(f_arr_v[-1]) + df_val, f_merger)
f_final_idx2 = np.argmin(np.abs(f_arr_v - f_final_hz))

# LAL's dphiXAS at f_final
dphiXAS_lal_val = dphi_xas_lal_df_v[f_final_idx2] / M_s
dphiT = float(phenomx_tidal_phase_derivative(theta_intrinsic, f_final_hz * M_s))

print(f"\nf_final = {f_final_hz:.4f} Hz")
print(f"dphiXAS (LAL): {dphiXAS_lal_val:+.10e}")
print(f"dphiT: {dphiT:+.10e}")
print(f"linb_step1: {linb_step1:.10f}")

# linb with LAL's dphiXAS
dphi_fmerger_lal = dphiXAS_lal_val + linb_step1 - dphiT
linb_final_lal = linb_step1 - dphi_fmerger_lal

# linb with ripple's dphiXAS
dphiXAS_rip_val = float(RipplePhaseDeriv(f_final_hz, theta_bbh, bbh_phase_coeffs) / M_s)
dphi_fmerger_rip = dphiXAS_rip_val + linb_step1 - dphiT
linb_final_rip = linb_step1 - dphi_fmerger_rip

print(f"linb_final (LAL dphiXAS): {linb_final_lal:.10f}")
print(f"linb_final (ripple dphiXAS): {linb_final_rip:.10f}")

# Now construct the NRTidalv3 phase with both linb values and compare
# The phase shift is: linb * f*M_s + lina + phifRef - 2*pi
# phifRef = -(Phase(f_ref) + linb*f_ref*M_s + lina - phiTfRef) + PI/4 + PI

Phase_fref = float(RipplePhaseXAS(f_ref, theta_bbh, bbh_phase_coeffs))
phiTfRef = float(phenomx_tidal_phase(theta_intrinsic, f_ref * M_s))

def make_strain(linb_final):
    """Construct NRTidalv3 strain with given linb_final."""
    phifRef = -(Phase_fref + linb_final * f_ref * M_s + lina - phiTfRef) + PI/4.0 + PI

    # Total phase
    Phase_arr = np.array([
        float(RipplePhaseXAS(float(f), theta_bbh, bbh_phase_coeffs))
        for f in f_arr_v
    ])

    f_Ms_arr = f_arr_v * M_s
    phase_shift_arr = linb_final * f_Ms_arr + lina + phifRef - 2.0 * PI

    # Tidal phase
    PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
    NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)

    psi_T_arr = np.zeros(len(f_arr_v))
    psi_QM_arr = np.zeros(len(f_arr_v))
    psi_SS_arr = np.zeros(len(f_arr_v))

    for i, f in enumerate(f_arr_v):
        fMs = f * M_s
        x = PI * fMs
        P_P = float(general_planck_taper(fMs, 1.15 * f_merger * M_s, 1.35 * f_merger * M_s))

        NRTidalv3_phase = float(get_tidal_phase(x, NRTidalv3_coeffs, PN_coeffs))
        PN_tidal = float(get_tidal_phase_PN(x, m1/(m1+m2), float(lambda1), float(lambda2), PN_coeffs))
        psi_T_arr[i] = NRTidalv3_phase * (1 - P_P) + PN_tidal * P_P
        psi_QM_arr[i] = float(get_qm_phase_correction(fMs, theta_intrinsic))
        psi_SS_arr[i] = float(get_spin_phase_correction(x ** (2.0/3.0), theta_intrinsic))

    total_phase = Phase_arr + phase_shift_arr - psi_T_arr - psi_QM_arr - psi_SS_arr

    # Amplitude
    A_T_arr = np.zeros(len(f_arr_v))
    A_P_arr = np.ones(len(f_arr_v))
    bbh_amp = np.abs(hp_xas_lal_v)

    for i, f in enumerate(f_arr_v):
        fMs = f * M_s
        x = PI * fMs
        x_23 = x ** (2.0/3.0)
        kappa = float(get_kappa(theta=theta_intrinsic))
        A_T_arr[i] = float(get_tidal_amplitude(x_23, theta_intrinsic, kappa, distance=100.0))
        A_P_arr[i] = float(get_planck_taper(jnp.array(f), f_merger))

    total_amp = A_P_arr * (bbh_amp + A_T_arr)

    return total_amp * np.exp(1j * total_phase)


# Generate strains
strain_lal_dphiXAS = make_strain(linb_final_lal)
strain_rip_dphiXAS = make_strain(linb_final_rip)

# Compute overlap loss with LAL NRTidalv3
def overlap_loss(h1, h2, freqs):
    """Simple overlap loss."""
    mask = freqs >= 20.0
    h1, h2 = h1[mask], h2[mask]
    freqs = freqs[mask]
    psd = freqs ** (-1)  # Simple 1/f weighting
    inner = np.sum(np.conj(h1) * h2 * psd)
    norm1 = np.sqrt(np.sum(np.conj(h1) * h1 * psd))
    norm2 = np.sqrt(np.sum(np.conj(h2) * h2 * psd))
    return 1 - np.abs(inner) / (norm1 * norm2)

loss_rip = overlap_loss(hp_nrtidal_lal_v, strain_rip_dphiXAS, f_arr_v)
loss_lal = overlap_loss(hp_nrtidal_lal_v, strain_lal_dphiXAS, f_arr_v)

print(f"\nOverlap loss with ripple dphiXAS: {loss_rip:.6e}")
print(f"Overlap loss with LAL dphiXAS:    {loss_lal:.6e}")
print(f"log10(ripple loss): {np.log10(loss_rip):.4f}")
print(f"log10(LAL loss): {np.log10(loss_lal):.4f}")
