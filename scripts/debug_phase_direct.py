#!/usr/bin/env python
"""Direct comparison: LAL's raw 22-mode phase vs ripple's Phase at a few frequencies.

If LAL's IMRPhenomX_Phase_22 = ripple's Phase, then the raw phase values should match
(up to the phifRef offset which is constant).
"""

import numpy as np
import lal
import lalsimulation as lalsim
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ripplegw.waveforms.IMRPhenomXAS import Phase as RipplePhase
from ripplegw.waveforms.IMRPhenomX_utils import PhenomX_phase_coeff_table
from ripplegw.constants import MTSUN

m1, m2 = 1.4, 1.35
chi1, chi2 = 0.02, 0.015
f_ref = 20.0
df = 1.0 / 128.0
theta_bbh = jnp.array([m1, m2, chi1, chi2])
bbh_phase_coeffs = PhenomX_phase_coeff_table
eta = m1 * m2 / (m1 + m2) ** 2
M_s = (m1 + m2) * MTSUN

# Generate LAL XAS
m1_kg = m1 * lal.MSUN_SI
m2_kg = m2 * lal.MSUN_SI
dist_SI = 100.0 * 1e6 * lal.PC_SI

approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")
hp, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0.0, 0.0, chi1, 0.0, 0.0, chi2,
    dist_SI, 0.0, 0.0, 0, 0, 0,
    df, 20.0, 4096.0, f_ref, None, approx,
)
hp_data = np.array(hp.data.data)
f_arr = np.arange(len(hp_data)) * df

# Extract LAL total phase at f_ref and 100 Hz
f_ref_idx = int(round(f_ref / df))
f_100_idx = int(round(100.0 / df))

phase_lal_fref = np.unwrap(np.angle(hp_data[f_ref_idx-2:f_ref_idx+3]))[2]
phase_lal_100 = np.unwrap(np.angle(hp_data[f_100_idx-2:f_100_idx+3]))[2]

# Ripple Phase at same frequencies
phase_rip_fref = float(RipplePhase(f_ref, theta_bbh, bbh_phase_coeffs))
phase_rip_100 = float(RipplePhase(100.0, theta_bbh, bbh_phase_coeffs))

print("Phase comparison (total phase, including 1/eta):")
print(f"  At f_ref = {f_ref} Hz:")
print(f"    LAL:   {phase_lal_fref:.6f}")
print(f"    Ripple: {phase_rip_fref:.6f}")
print(f"    Difference: {phase_lal_fref - phase_rip_fref:.6f}")

print(f"\n  At f = 100 Hz:")
print(f"    LAL:   {phase_lal_100:.6f}")
print(f"    Ripple: {phase_rip_100:.6f}")
print(f"    Difference: {phase_lal_100 - phase_rip_100:.6f}")

# The LAL total phase = (1/eta) * phi_22 + phifRef
# The ripple Phase = (1/eta) * phi_22
# So: LAL total - ripple Phase = phifRef (constant)

# If the difference is NOT constant, then the phi_22 functions differ.
delta_fref = phase_lal_fref - phase_rip_fref
delta_100 = phase_lal_100 - phase_rip_100

print(f"\n  Delta at f_ref: {delta_fref:.6f}")
print(f"  Delta at 100 Hz: {delta_100:.6f}")
print(f"  Delta difference: {delta_100 - delta_fref:.6f}")

if abs(delta_100 - delta_fref) < 0.001:
    print("\n  => phi_22 matches (constant offset = phifRef)")
else:
    print(f"\n  => phi_22 DOES NOT MATCH! Difference = {delta_100 - delta_fref:.6f}")
    print("  This confirms the inspiral phase formulas differ between LAL and ripple.")

# Compute the phase derivative at 100 Hz
# LAL: d(Phase_total)/dMf at 100 Hz
dphi_lal_100 = (phase_lal_100 - phase_lal_fref) / ((100.0 - f_ref) * M_s)
dphi_rip_100 = (phase_rip_100 - phase_rip_fref) / ((100.0 - f_ref) * M_s)

print(f"\n  Approximate d(Phase)/dMf at 100 Hz:")
print(f"    LAL:   {dphi_lal_100:.6f}")
print(f"    Ripple: {dphi_rip_100:.6f}")
print(f"    Difference: {dphi_lal_100 - dphi_rip_100:.6f}")
