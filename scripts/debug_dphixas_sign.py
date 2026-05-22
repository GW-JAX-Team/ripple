#!/usr/bin/env python
"""Quick test: what happens if we use the correct sign for dphiXAS?"""

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
)
from ripplegw.constants import MTSUN, PI

# Test parameters
m1, m2 = 1.4, 1.35
chi1, chi2 = 0.02, 0.015
lambda1, lambda2 = 400.0, 300.0
f_ref = 20.0
df = 1.0 / 128.0
T = 128.0

Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(jnp.array([lambda1, lambda2, m1, m2]))
M_s = (m1 + m2) * MTSUN
theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
theta_bbh = jnp.array([m1, m2, chi1, chi2])
bbh_phase_coeffs = PhenomX_phase_coeff_table

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
f_final = min(f_ref + (2037.5 - f_ref), f_merger)  # approximate f_final

# Current ripple dphiXAS (using PhaseDerivative)
from ripplegw.waveforms.IMRPhenomXAS import PhaseDerivative as RipplePhaseDeriv
dphiXAS_ripple = float(RipplePhaseDeriv(f_final, theta_bbh, bbh_phase_coeffs) / M_s)

# LAL-like dphiXAS (positive, from the complex strain extraction)
# At f_final ~ 2037 Hz, LAL gives d(Phase_total)/df = +0.0033
# dphi_22/dMf = eta/M_s * d(Phase_total)/df = eta/M_s * 0.0033
dphiXAS_lal = float(eta) / M_s * 0.003307364  # from earlier diagnostic

dphiT = float(phenomx_tidal_phase_derivative(theta_intrinsic, f_final * M_s))

print("dphiXAS comparison:")
print(f"  Ripple (PhaseDerivative/M_s): {dphiXAS_ripple:.10f}")
print(f"  LAL-like (eta/M_s * d(Phi)/df): {dphiXAS_lal:.10f}")
print(f"  dphiT: {dphiT:.10f}")

# Compute linb_final with both
dphi_fmerger_ripple = dphiXAS_ripple + linb_step1 - dphiT
linb_final_ripple = linb_step1 - dphi_fmerger_ripple

dphi_fmerger_lal = dphiXAS_lal + linb_step1 - dphiT
linb_final_lal = linb_step1 - dphi_fmerger_lal

print(f"\nlinb computation:")
print(f"  linb_step1: {linb_step1:.10f}")
print(f"  With ripple dphiXAS:")
print(f"    dphi_fmerger: {dphi_fmerger_ripple:.10f}")
print(f"    linb_final: {linb_final_ripple:.10f}")
print(f"  With LAL-like dphiXAS:")
print(f"    dphi_fmerger: {dphi_fmerger_lal:.10f}")
print(f"    linb_final: {linb_final_lal:.10f}")

# The key difference in linb_final:
print(f"\nlinb_final difference: {linb_final_lal - linb_final_ripple:.10f}")
# This difference would cause a phase shift of:
# delta_phase = (linb_final_lal - linb_final_ripple) * (f - f_ref) * M_s
# At 100 Hz: delta_phase = d_linb * 80 * M_s

d_linb = linb_final_lal - linb_final_ripple
for f_check in [50, 100, 200, 500, 1000]:
    delta_phase = float(d_linb) * (f_check - f_ref) * M_s
    print(f"  Phase shift at {f_check} Hz: {delta_phase:.6e} rad")

# Compare with the actual residual
# From earlier: residual at 100 Hz = -1.15e-4 rad
# The predicted phase shift from linb difference: {d_linb * 80 * M_s:.6e}
print(f"\n  Actual residual at 100 Hz: -1.15e-4 rad")
print(f"  Predicted from linb: {float(d_linb) * 80 * M_s:.6e} rad")
