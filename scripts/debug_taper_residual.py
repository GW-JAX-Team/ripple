#!/usr/bin/env python
"""Check if the residual matches the taper blending difference."""

import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from ripplegw.waveforms.IMRPhenomXAS import Phase as RipplePhaseXAS
from ripplegw.waveforms.IMRPhenomX_utils import PhenomX_phase_coeff_table
from ripplegw.waveforms.NRTidalv3_utils import (
    phenomx_tidal_phase,
    _get_merger_frequency,
    get_tidal_phase,
    get_tidal_phase_PN,
    get_tidalphasePN_coeffs,
    get_NRTidalv3_coefficients,
    general_planck_taper,
)
from ripplegw.constants import MTSUN, PI

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

# Compute merger frequencies
f_merger_ripple = float(_get_merger_frequency(theta_intrinsic))
print(f"f_merger (ripple): {f_merger_ripple:.4f} Hz")
print(f"Taper start (1.15 * f_merger): {1.15 * f_merger_ripple:.4f} Hz")

# Get LAL merger frequency
# We need to call LAL's XLALSimNRTunedTidesMergerFrequency_v3
# This is a C function, but we can call it through the Python bindings
# Actually, LAL computes the merger frequency during waveform generation.
# Let me check if the ripple and LAL merger frequencies match by looking at where the taper starts.

# For LAL, the taper starts at 1.15 * f_merger where f_merger is computed by
# XLALSimNRTunedTidesMergerFrequency_v3. Let's see if this matches ripple.

# Compute coefficients
PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)

# Compare taper at various frequencies
print(f"\nTaper comparison:")
print(f"{'f(Hz)':>8} {'Ripple taper':>14}")
for f_check in [1000, 1500, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800]:
    fMs = f_check * M_s
    taper_rip = float(general_planck_taper(fMs, 1.15 * f_merger_ripple * M_s, 1.35 * f_merger_ripple * M_s))
    print(f"{f_check:8.0f} {taper_rip:14.6f}")

# Generate LAL and ripple waveforms to compare the actual taper behavior
m1_kg = m1 * lal.MSUN_SI
m2_kg = m2 * lal.MSUN_SI
dist_SI = 100.0 * 1e6 * lal.PC_SI

# LAL NRTidalv3
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
    df, 20.0, 4096.0, f_ref, laldict, approx_nrtidal,
)

# LAL XAS
approx_xas = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")
hp_xas_lal, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0.0, 0.0, chi1, 0.0, 0.0, chi2,
    dist_SI, 0.0, 0.0, 0, 0, 0,
    df, 20.0, 4096.0, f_ref, None, approx_xas,
)

hp_nrtidal_lal_data = np.array(hp_nrtidal_lal.data.data)
hp_xas_lal_data = np.array(hp_xas_lal.data.data)

# Ripple waveforms
from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3_hphc
from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc

fs = jnp.arange(len(hp_nrtidal_lal_data)) * df

params_nrtidal = jnp.array([Mc, eta, chi1, chi2, lambda_tilde, delta_lambda_tilde, 100.0, 0.0, 0.0, 0.0])
params_xas = jnp.array([Mc, eta, chi1, chi2, 100.0, 0.0, 0.0, 0.0])

hp_nrtidal_rip, _ = gen_IMRPhenomXAS_NRTidalv3_hphc(fs, params_nrtidal, f_ref)
hp_xas_rip, _ = gen_IMRPhenomXAS_hphc(fs, params_xas, f_ref)

# Non-zero mask
nonzero = np.where((np.abs(hp_nrtidal_lal_data) > 0) & (np.abs(hp_xas_lal_data) > 0))[0]
f_arr = fs[nonzero]

# Phase differences
phase_nrtidal_lal = np.unwrap(np.angle(hp_nrtidal_lal_data[nonzero]))
phase_xas_lal = np.unwrap(np.angle(hp_xas_lal_data[nonzero]))
phase_nrtidal_rip = np.unwrap(np.angle(np.array(hp_nrtidal_rip)[nonzero]))
phase_xas_rip = np.unwrap(np.angle(np.array(hp_xas_rip)[nonzero]))

diff_lal = phase_nrtidal_lal - phase_xas_lal
diff_rip = phase_nrtidal_rip - phase_xas_rip

f_ref_idx = np.argmin(np.abs(f_arr - f_ref))
diff_lal_aligned = diff_lal - diff_lal[f_ref_idx]
diff_rip_aligned = diff_rip - diff_rip[f_ref_idx]

residual = diff_lal_aligned - diff_rip_aligned

# Now compute what the phase difference SHOULD be based on the tidal phase alone
# (ignoring the linb/tidal alignment)
print(f"\n--- Tidal phase comparison ---")
for f_check in [100, 200, 500, 1000, 1500, 2000]:
    idx = np.argmin(np.abs(f_arr - f_check))
    f_val = float(f_arr[idx])
    fMs = f_val * M_s
    x_val = PI * fMs

    # LAL tidal phase (extracted from phase difference + linb correction)
    tidal_lal = float(diff_lal_aligned[idx])

    # Ripple tidal phase
    tidal_rip = float(diff_rip_aligned[idx])

    # Pure NRTidalv3 phase (no linb correction, no QM, no SS)
    NRTidalv3_phase = float(get_tidal_phase(x_val, NRTidalv3_coeffs, PN_coeffs))

    # phenomx_tidal_phase (includes spin terms)
    phenomx_tidal = float(phenomx_tidal_phase(theta_intrinsic, fMs))

    print(f"  f={f_val:7.2f}: LAL diff={tidal_lal:+.6e}, Rip diff={tidal_rip:+.6e}, "
          f"NRTidalv3={NRTidalv3_phase:+.6e}, phenomx_tidal={phenomx_tidal:+.6e}")

# The difference between LAL and ripple phase differences should be purely from
# the linb alignment (since the tidal phases match).
# Let's check if the residual is constant in the non-tapered region.

print(f"\n--- Residual analysis ---")
print(f"{'f(Hz)':>8} {'Residual':>14} {'Taper (rip)':>14}")
for f_check in [100, 200, 500, 1000, 1500, 2000, 2100, 2200, 2300, 2400]:
    idx = np.argmin(np.abs(f_arr - f_check))
    f_val = float(f_arr[idx])
    fMs = f_val * M_s
    taper = float(general_planck_taper(fMs, 1.15 * f_merger_ripple * M_s, 1.35 * f_merger_ripple * M_s))
    res = float(residual[idx])
    print(f"{f_val:8.2f} {res:+14.6e} {taper:14.6f}")
