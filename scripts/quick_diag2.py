#!/usr/bin/env python
"""Check dphiXAS_rip vs dphiXAS_LAL at the correct f_final."""
import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.waveforms.IMRPhenomXAS import Phase as RipplePhase, PhaseDerivative
from ripplegw.waveforms.IMRPhenomX_utils import PhenomX_phase_coeff_table, calc_phaseatpeak, get_cutoff_fMs
from ripplegw.waveforms.NRTidalv3_utils import _get_merger_frequency, phenomx_tidal_phase_derivative, phenomx_tidal_phase
from ripplegw.conversions import ms_to_Mc_eta
from ripplegw.constants import MTSUN, PI

m1, m2 = 1.4, 1.4
chi1, chi2 = 0.0, 0.0
lambda1, lambda2 = 1000.0, 1000.0
T = 128.0
df = 1.0 / T
f_ref = 20.0
f_max = 4096.0

M_s = (m1 + m2) * MTSUN
Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
theta_bbh = jnp.array([m1, m2, chi1, chi2])
bbh_phase_coeffs = PhenomX_phase_coeff_table

f_merger = float(_get_merger_frequency(theta_intrinsic))

# Ripple frequency grid
f_sampling = 2 * f_max
delta_t = 1.0 / f_sampling
tlen = int(round(T / delta_t))
freqs_all = np.fft.rfftfreq(tlen, delta_t)
mask = (freqs_all > f_ref) & (freqs_all < f_max)
f_arr = freqs_all[mask]

# f_final that ripple uses
f_final_rip = min(f_arr[-1] + df, f_merger)
print(f"f_merger = {f_merger:.6f} Hz")
print(f"f_final_rip = {f_final_rip:.6f} Hz, Mf_final = {f_final_rip * M_s:.8f}")

# Generate LAL XAS waveform
m1_kg, m2_kg = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
dist_SI = 100e6 * lal.PC_SI
approx_xas = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")
hp_xas, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0, 0, chi1, 0, 0, chi2,
    dist_SI, 0, 0, 0, 0, 0,
    df, 20.0, f_max, f_ref, None, approx_xas,
)
f_arr_lal = np.arange(len(hp_xas.data.data)) * df
xas_data = np.array(hp_xas.data.data)

# Extract LAL's d(Phase)/df at f_final_rip using backward secant
idx_final = int(round(f_final_rip / df))  # index corresponding to f_final_rip in LAL array
print(f"idx_final = {idx_final}, f_arr_lal[idx_final] = {f_arr_lal[idx_final]:.8f}")

h_final = xas_data[idx_final]
h_prev = xas_data[idx_final - 1]
print(f"|h_final| = {np.abs(h_final):.6e}, |h_prev| = {np.abs(h_prev):.6e}")

# d(Phase)/df via backward secant: Im(conj(h_prev)*h_final) / (|h_prev||h_final| * df)
# But this gives the phase difference, not the derivative!
# Actually: Phase(f) - Phase(f-df) ≈ df * d(Phase)/df
# and h = A * exp(i*Phase), so Phase(f) - Phase(f-df) = angle(h_final * conj(h_prev))
# = Im(log(h_final * conj(h_prev))) = arctan(Im(h_final*conj(h_prev)) / Re(h_final*conj(h_prev)))
# This is only valid if the phase difference < pi per bin, which is definitely true here.

phase_diff = np.angle(h_final * np.conj(h_prev))
# But there's a sign convention issue - Phase_LAL could be negative
# The derivative is: (Phase_final - Phase_prev) / df
# Since LAL uses h = A*exp(-i*phi), we need to extract the phase as -angle(h)
# Actually let's just use: angle(h_final) - angle(h_prev) (adjusting for wraps)
phi_final_lal = np.angle(h_final)
phi_prev_lal = np.angle(h_prev)
dphi_df_lal_v1 = np.unwrap(np.array([phi_prev_lal, phi_final_lal]))[1] - np.unwrap(np.array([phi_prev_lal, phi_final_lal]))[0]
dphi_df_lal_v1 /= df

# Better: use the formula d(Phase)/df = Im(conj(h) * dh/df) / |h|^2
# dh/df ≈ (h_final - h_prev) / df (backward difference)
dh_df = (h_final - h_prev) / df
dphi_df_lal = np.imag(np.conj(h_final) * dh_df) / np.abs(h_final)**2

dphiXAS_lal = dphi_df_lal / M_s  # convert to d/dMf
print(f"\nd(Phase)/df at f_final from LAL (Im method): {dphi_df_lal:.10e}")
print(f"d(Phase)/dMf from LAL: {dphiXAS_lal:.10e}")
print(f"d(Phase)/df at f_final from angle method: {dphi_df_lal_v1:.10e}")

# Compare ripple
dphiXAS_rip_secant = float(
    (RipplePhase(f_final_rip, theta_bbh, bbh_phase_coeffs) -
     RipplePhase(f_final_rip - df, theta_bbh, bbh_phase_coeffs)) / (df * M_s)
)
dphiXAS_rip_analytic = float(PhaseDerivative(f_final_rip, theta_bbh, bbh_phase_coeffs) / M_s)

print(f"\ndphiXAS_rip (backward secant): {dphiXAS_rip_secant:.10e}")
print(f"dphiXAS_rip (analytic):        {dphiXAS_rip_analytic:.10e}")
print(f"dphiXAS_lal (from waveform):   {dphiXAS_lal:.10e}")
print(f"Diff rip_secant - lal:  {dphiXAS_rip_secant - dphiXAS_lal:.6e}")
print(f"Diff rip_analytic - lal: {dphiXAS_rip_analytic - dphiXAS_lal:.6e}")

# Also check if the assembled phases match at f_final
Phase_rip_at_ffinal = float(RipplePhase(f_final_rip, theta_bbh, bbh_phase_coeffs))
Phase_lal_at_ffinal = -np.angle(h_final)  # LAL uses h = A*exp(-i*Phase)

print(f"\nPhase_rip at f_final: {Phase_rip_at_ffinal:.10f}")
print(f"Phase_lal at f_final (=-angle(h)): {Phase_lal_at_ffinal:.10f}")
print(f"Diff (but ignoring global constant): {Phase_rip_at_ffinal - Phase_lal_at_ffinal:.10f}")

# Compute dphiT at f_final
dphiT = float(phenomx_tidal_phase_derivative(theta_intrinsic, f_final_rip * M_s))
print(f"\ndphiT at f_final: {dphiT:.10e}")

# linb_final comparison
linb_final_rip = dphiT - dphiXAS_rip_secant
linb_final_lal = dphiT - dphiXAS_lal

print(f"\nlinb_final_rip: {linb_final_rip:.10e}")
print(f"linb_final_lal: {linb_final_lal:.10e}")
print(f"delta_linb: {linb_final_rip - linb_final_lal:.6e}")

# The critical question: does ripple's assembled phase have a different slope at f_final
# compared to LAL? Let's check phase values at many points near f_final to see if there's
# a systematic linear offset between Phase_rip and Phase_LAL.

# Build full phase arrays
f_check = np.array([20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, f_final_rip])
for fc in f_check:
    idx = int(round(fc / df))
    if 0 <= idx < len(xas_data) and np.abs(xas_data[idx]) > 0:
        Phase_rip_val = float(RipplePhase(fc, theta_bbh, bbh_phase_coeffs))
        Phase_lal_val = -np.angle(xas_data[idx])  # LAL: h = A*exp(-i*Phase)
        print(f"f={fc:.1f} Hz: Phase_rip={Phase_rip_val:.6f}, -angle(h_LAL)={Phase_lal_val:.6f}, diff={Phase_rip_val - Phase_lal_val:.6f}")
