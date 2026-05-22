#!/usr/bin/env python
"""Check LAL XAS phase structure."""

import numpy as np
import lal
import lalsimulation as lalsim

m1, m2 = 1.4, 1.35
chi1, chi2 = 0.02, 0.015
df = 1.0 / 128.0

m1_kg = m1 * lal.MSUN_SI
m2_kg = m2 * lal.MSUN_SI
dist_SI = 100.0 * 1e6 * lal.PC_SI
f_ref = 20.0

approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")
hp, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0.0, 0.0, chi1, 0.0, 0.0, chi2,
    dist_SI, 0.0, 0.0, 0, 0, 0,
    df, 20.0, 4096.0, f_ref, None, approximant,
)
hp_data = np.array(hp.data.data)
f_arr = np.arange(len(hp_data)) * df

nonzero = np.where(np.abs(hp_data) > 0)[0]
hp_data = hp_data[nonzero[0]:nonzero[-1]+1]
f_arr = f_arr[nonzero[0]:nonzero[-1]+1]

# Check phase at several frequencies
phase_raw = np.angle(hp_data)
phase_unwrapped = np.unwrap(phase_raw)

print("LAL XAS phase analysis:")
print(f"  Frequency range: {f_arr[0]:.2f} - {f_arr[-1]:.2f} Hz")
print(f"  Number of bins: {len(f_arr)}")
print(f"  df: {df:.6f} Hz")

for f_check in [20, 50, 100, 200, 500, 1000, 1500, 2000, 2441]:
    idx = np.argmin(np.abs(f_arr - f_check))
    f_val = f_arr[idx]
    h_val = hp_data[idx]
    p_raw = np.angle(h_val)
    p_unwrap = phase_unwrapped[idx]
    amp = np.abs(h_val)
    print(f"  f={f_val:7.2f}: amp={amp:.6e}, angle={p_raw:+.6f}, unwrapped={p_unwrap:+.6f}")

# Check if the phase is actually accumulated (monotonic) or wrapped
print(f"\n  Phase change from f=20 to f=2441:")
print(f"    Raw: {phase_raw[-1] - phase_raw[0]:+.6f}")
print(f"    Unwrapped: {phase_unwrapped[-1] - phase_unwrapped[0]:+.6f}")

# Check the derivative from the complex values directly
# d(phi)/df = Im(h* * dh/df) / |h|^2
# Using finite difference for dh/df
h_conj = np.conj(hp_data)
dh_df = np.zeros_like(hp_data)
for i in range(1, len(hp_data) - 1):
    dh_df[i] = (hp_data[i+1] - hp_data[i-1]) / (2 * df)

dphi_df = np.imag(h_conj * dh_df) / (np.abs(hp_data) ** 2)

print(f"\n  d(phi)/df from complex values:")
for f_check in [20, 50, 100, 200, 500, 1000, 1500, 2000, 2037]:
    idx = np.argmin(np.abs(f_arr - f_check))
    if 1 <= idx < len(dphi_df) - 1:
        print(f"    f={f_arr[idx]:7.2f}: d(phi)/df = {dphi_df[idx]:+.6e}")
