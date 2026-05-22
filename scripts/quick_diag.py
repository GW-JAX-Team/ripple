#!/usr/bin/env python
"""Quick diagnostic: compare dphiXAS between ripple and LAL."""

import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.conversions import ms_to_Mc_eta
from ripplegw.waveforms.IMRPhenomXAS import Phase as RipplePhase, PhaseDerivative
from ripplegw.waveforms.IMRPhenomX_utils import PhenomX_phase_coeff_table, calc_phaseatpeak, get_cutoff_fMs
from ripplegw.waveforms.NRTidalv3_utils import _get_merger_frequency, phenomx_tidal_phase_derivative
from ripplegw.constants import MTSUN, PI

m1, m2 = 1.4, 1.4
chi1, chi2 = 0.0, 0.0
lambda1, lambda2 = 1000.0, 1000.0
T = 128.0
df = 1.0 / T
f_ref = 20.0

M_s = (m1 + m2) * MTSUN
Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
theta_bbh = jnp.array([m1, m2, chi1, chi2])
bbh_phase_coeffs = PhenomX_phase_coeff_table

f_merger = float(_get_merger_frequency(theta_intrinsic))
f_max = 4096.0
# Compute f_final using the test grid
f_sampling = 2 * f_max
delta_t = 1.0 / f_sampling
tlen = int(round(T / delta_t))
freqs = np.fft.rfftfreq(tlen, delta_t)
mask = (freqs > f_ref) & (freqs < f_max)
f_arr = freqs[mask]
f_final_rip = f_arr[-1] + df  # ripple's f_final
f_final_rip = min(f_final_rip, f_merger)

print(f"f_merger = {f_merger:.4f} Hz")
print(f"f_arr[-1] = {f_arr[-1]:.8f} Hz")
print(f"f_final_rip = {f_final_rip:.8f} Hz, Mf = {f_final_rip * M_s:.8f}")

# Generate LAL XAS waveform to extract dphiXAS numerically
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
nz = np.where(np.abs(xas_data) > 0)[0]
print(f"LAL XAS valid range: {f_arr_lal[nz[0]]:.4f} - {f_arr_lal[nz[-1]]:.4f} Hz")

# Extract LAL's d(Phase)/dMf at f_final using central differences
# d(Phase)/df = Im(conj(h) * dh/df) / |h|^2
# d(Phase)/dMf = d(Phase)/df / M_s
f_final_lal = f_arr_lal[nz[-1]]  # last nonzero LAL frequency
print(f"LAL last nonzero freq = {f_final_lal:.4f} Hz")

# Use backward secant to estimate d(Phase)/df at f_final from LAL
idx_final = nz[-1]
# Make sure we have indices idx_final-1 and idx_final
if idx_final >= 1 and np.abs(xas_data[idx_final-1]) > 0:
    h_curr = xas_data[idx_final]
    h_prev = xas_data[idx_final-1]
    # Phase derivative: backward finite difference
    dphase_lal_df = np.imag(np.conj(h_prev) * h_curr) / (np.abs(h_prev) * np.abs(h_curr)) / df
else:
    dphase_lal_df = np.nan

dphiXAS_lal_at_final = dphase_lal_df / M_s
print(f"\nd(Phase)/df at LAL last freq (backward secant): {dphase_lal_df:.10e}")
print(f"d(Phase)/dMf = dphiXAS_LAL: {dphiXAS_lal_at_final:.10e}")

# Ripple's dphiXAS at f_final (backward secant)
f_final_val = float(f_final_rip)
dphiXAS_rip_secant = float(
    (RipplePhase(f_final_val, theta_bbh, bbh_phase_coeffs) -
     RipplePhase(f_final_val - df, theta_bbh, bbh_phase_coeffs)) / (df * M_s)
)
dphiXAS_rip_analytic = float(PhaseDerivative(f_final_val, theta_bbh, bbh_phase_coeffs) / M_s)

print(f"\ndphiXAS_rip (backward secant): {dphiXAS_rip_secant:.10e}")
print(f"dphiXAS_rip (analytic): {dphiXAS_rip_analytic:.10e}")
print(f"Diff (secant - analytic): {dphiXAS_rip_secant - dphiXAS_rip_analytic:.4e}")
print(f"Diff (LAL - rip_secant): {dphiXAS_lal_at_final - dphiXAS_rip_secant:.4e}")

# Compute dphiT
dphiT = float(phenomx_tidal_phase_derivative(theta_intrinsic, f_final_val * M_s))
print(f"\ndphiT (at f_final): {dphiT:.10e}")

# Ripple's linb_final
delta = float(jnp.sqrt(1.0 - 4.0 * float(eta)))
mm1 = 0.5 * (1.0 + delta)
mm2 = 0.5 * (1.0 - delta)
StotR = float((mm1**2 + mm2**2)**(-1.0) * (mm1**2 * chi1 + mm2**2 * chi2))
chia = chi1 - chi2

lina, linb_init, psi4tostrain = calc_phaseatpeak(float(eta), StotR, chia, delta)
fMs_RD, fMs_damp, _, _ = get_cutoff_fMs(m1, m2, chi1, chi2)
dphi22Ref = float(jax.grad(RipplePhase)((fMs_RD - fMs_damp) / M_s, theta_bbh, bbh_phase_coeffs) / M_s)
linb_BBH = float(linb_init) - dphi22Ref - 2.0 * PI * (500.0 + float(psi4tostrain))

linb_final_rip = dphiT - dphiXAS_rip_secant
linb_final_lal_est = dphiT - dphiXAS_lal_at_final
delta_linb = linb_final_rip - linb_final_lal_est

print(f"\nlinb_BBH (ripple): {linb_BBH:.10e}")
print(f"linb_final_rip (= dphiT - dphiXAS_rip): {linb_final_rip:.10e}")
print(f"linb_final_LAL_est (= dphiT - dphiXAS_LAL): {linb_final_lal_est:.10e}")
print(f"delta_linb = {delta_linb:.10e}")

# Predict overlap loss from linear phase error
# Phase error at f: delta_linb * (f - f_ref) * M_s
print(f"\nPredicted phase errors from delta_linb:")
for f_check in [50, 100, 200, 500, 1000]:
    dphase = delta_linb * (f_check - f_ref) * M_s
    print(f"  {f_check} Hz: {dphase:.4e} rad")

# Key: compare ripple Phase at two nearby points
f_test = float(f_arr[-1])  # 4095.992...
print(f"\nPhase comparison at f = {f_test:.6f} Hz:")
print(f"  Phase_rip(f) = {float(RipplePhase(f_test, theta_bbh, bbh_phase_coeffs)):.10e}")
print(f"  Phase_rip(f-df) = {float(RipplePhase(f_test - df, theta_bbh, bbh_phase_coeffs)):.10e}")
print(f"  Phase_rip(f) - Phase_rip(f-df) = {float(RipplePhase(f_test, theta_bbh, bbh_phase_coeffs) - RipplePhase(f_test - df, theta_bbh, bbh_phase_coeffs)):.10e}")
print(f"  (divided by df * M_s) = {float(RipplePhase(f_test, theta_bbh, bbh_phase_coeffs) - RipplePhase(f_test - df, theta_bbh, bbh_phase_coeffs)) / (df * M_s):.10e}")
