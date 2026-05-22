#!/usr/bin/env python
"""Find the source of the -75.26 constant offset."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ripplegw.waveforms.IMRPhenomXAS import (
    get_inspiral_phase,
    get_intermediate_raw_phase,
    get_mergerringdown_raw_phase,
)
from ripplegw.waveforms.IMRPhenomX_utils import (
    PhenomX_phase_coeff_table,
    get_cutoff_fMs,
)
from ripplegw.constants import MTSUN, PI

m1, m2 = 1.4, 1.35
chi1, chi2 = 0.02, 0.015
theta_bbh = jnp.array([m1, m2, chi1, chi2])
M_s = (m1 + m2) * MTSUN
eta = m1 * m2 / (m1 + m2) ** 2
bbh_phase_coeffs = PhenomX_phase_coeff_table

fMs_RD, fMs_damp, fMs_MECO, fMs_ISCO = get_cutoff_fMs(m1, m2, chi1, chi2)
fMs_IMmatch = 0.6 * (0.5 * fMs_RD + fMs_ISCO)
fMs_INmatch = fMs_MECO
deltafMs = (fMs_IMmatch - fMs_INmatch) * 0.03
f1_Ms = fMs_INmatch - 1.0 * deltafMs
f2_Ms = fMs_IMmatch + 0.5 * deltafMs

# Compute matching coefficients
phi_Ins_match_f1, dphi_Ins_match_f1 = jax.value_and_grad(get_inspiral_phase)(
    f1_Ms, theta_bbh, bbh_phase_coeffs
)
phi_MRD_match_f2, dphi_MRD_match_f2 = jax.value_and_grad(
    get_mergerringdown_raw_phase, has_aux=True
)(f2_Ms, theta_bbh, bbh_phase_coeffs)
phi_MRD_match_f2_val, (cL, CV_phase_RD0) = get_mergerringdown_raw_phase(
    f2_Ms, theta_bbh, bbh_phase_coeffs
)

phi_Int_match_f1, dphi_Int_match_f1 = jax.value_and_grad(
    get_intermediate_raw_phase
)(f1_Ms, theta_bbh, bbh_phase_coeffs, dphi_Ins_match_f1, CV_phase_RD0, cL)

alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1
alpha0 = float(phi_Ins_match_f1) - float(phi_Int_match_f1) - float(alpha1) * float(f1_Ms)

phi_Int_func = lambda fM_s_: (
    get_intermediate_raw_phase(
        fM_s_, theta_bbh, bbh_phase_coeffs, dphi_Ins_match_f1, CV_phase_RD0, cL
    )
    + alpha1 * fM_s_
    + alpha0
)

phi_Int_match_f2, dphi_Int_match_f2 = jax.value_and_grad(phi_Int_func)(f2_Ms)
beta1 = dphi_Int_match_f2 - dphi_MRD_match_f2

print("Matching coefficients:")
print(f"  alpha1 = {float(alpha1):.10f}")
print(f"  alpha0 = {float(alpha0):.10f}")
print(f"  beta1  = {float(beta1):.10f}")
print(f"  dphi_Ins_match_f1  = {float(dphi_Ins_match_f1):.10f}")
print(f"  dphi_Int_match_f1  = {float(dphi_Int_match_f1):.10f}")
print(f"  dphi_Int_match_f2  = {float(dphi_Int_match_f2):.10f}")
print(f"  dphi_MRD_match_f2  = {float(dphi_MRD_match_f2):.10f}")

# The constant offset is -75.25803
# Check if it matches any coefficient:
target = -75.25803
print(f"\n  Target offset: {target:.6f}")
print(f"  alpha1 matches? {abs(float(alpha1) - target) < 0.001}")
print(f"  beta1 matches?  {abs(float(beta1) - target) < 0.001}")

# Let's check: at f in the INSPIRAL region, what is the difference?
# Both LAL and ripple should use the same get_inspiral_phase formula.
# The difference should be 0.

# But the diagnostic showed diff = -75.26 at 100-1500 Hz (all inspiral region).
# This is strange because the inspiral phase formula is identical.

# Wait - at 1500 Hz, fM_s = 0.02032, which is very close to f1_Ms = 0.02077.
# Maybe there's a region boundary issue?

print("\n--- Region check ---")
for f_check in [100, 500, 1000, 1400, 1500, 1600, 2000, 2500, 3000]:
    fM_s = f_check * M_s
    in_inspiral = fM_s < f1_Ms
    in_intermediate = f1_Ms <= fM_s < f2_Ms
    in_mrd = fM_s >= f2_Ms
    region = "inspiral" if in_inspiral else ("intermediate" if in_intermediate else "MRD")
    print(f"  f={f_check} Hz: fM_s={fM_s:.8f}, f1_Ms={f1_Ms:.8f}, region={region}")

# Check the derivative at 1500 Hz more carefully
f_check = 1500
fM_s = f_check * M_s
print(f"\n--- Detailed check at {f_check} Hz ---")
print(f"  fM_s = {fM_s:.10f}")
print(f"  f1_Ms = {f1_Ms:.10f}")
print(f"  fM_s < f1_Ms? {fM_s < f1_Ms}")

# Inspiral derivative at 1500 Hz
dphi_ins_1500 = float(jax.grad(get_inspiral_phase)(fM_s, theta_bbh, bbh_phase_coeffs))
print(f"  d(get_inspiral_phase)/dMf at 1500 Hz: {dphi_ins_1500:.6f}")

# This should be the LAL value (since LAL also uses the inspiral ansatz here).
# The ripple diagnostic showed dphi_raw/dMf = 58.01 at 1500 Hz.
# If LAL's is 133.27, the difference is -75.26.
# But both should use the same inspiral formula!

# Unless... the LAL diagnostic was wrong?
# The LAL dphi_22/dMf was computed as: eta/M_s * d(Phase_total)/df
# where d(Phase_total)/df was extracted from the complex strain.

# If LAL's Phase_total uses a different 1/eta convention than expected,
# the extracted dphi_22/dMf would be wrong.

# Let me check: what if LAL's 1/eta is applied differently?
# LAL: Phase_total = (1/eta) * phi_22 + phifRef
# So d(Phase_total)/df = (1/eta) * dphi_22/dMf * M_s
# => dphi_22/dMf = eta / M_s * d(Phase_total)/df

# But what if LAL's phi_22 already includes 1/eta?
# Then: Phase_total = phi_22_already_scaled + phifRef
# d(Phase_total)/df = dphi_22_already_scaled/dMf * M_s
# = (1/eta) * dphi_22_raw/dMf * M_s
# => dphi_22_raw/dMf = eta / M_s * d(Phase_total)/df

# This is the same formula! So the extraction should be correct.

# Let me check: what if the LAL inspiral phase uses a DIFFERENT formula than ripple?
# Both use TaylorF2 terms, but maybe LAL has additional calibration terms?

# Actually, looking at the LAL IMRPhenomX inspiral phase, there might be
# PhenomX-specific calibration terms (sigma coefficients) that ripple
# implements differently.

# Let me check the sigma terms in the ripple inspiral phase:
print("\n--- Checking sigma terms ---")
