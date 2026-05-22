#!/usr/bin/env python
"""Direct comparison of LAL vs ripple 22-mode phase derivative (without 1/eta)."""

import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

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

print(f"Matching frequencies: f1_Ms={f1_Ms:.8f}, f2_Ms={f2_Ms:.8f}")
print(f"fMs_MECO={fMs_MECO:.8f}, fMs_IMmatch={fMs_IMmatch:.8f}")

# Generate LAL XAS waveform and extract the 22-mode phase derivative
# at specific frequencies using the complex strain
m1_kg = m1 * lal.MSUN_SI
m2_kg = m2 * lal.MSUN_SI
dist_SI = 100.0 * 1e6 * lal.PC_SI
f_ref = 20.0
df = 1.0 / 128.0

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

# Compute d(Phase_total)/df from complex values
h_conj = np.conj(hp_data)
dh_df = np.zeros_like(hp_data, dtype=complex)
for i in range(2, len(hp_data) - 2):
    dh_df[i] = (-hp_data[i+2] + 8*hp_data[i+1] - 8*hp_data[i-1] + hp_data[i-2]) / (12 * df)

dphi_total_df = np.imag(h_conj * dh_df) / (np.abs(hp_data) ** 2)

# This gives d(Phase_total)/df = (1/eta) * dphi_22/dMf * M_s * df/d(f*M_s)
# Wait, let me think more carefully.
# Phase_total = (1/eta) * phi_22(f*M_s) + phifRef
# d(Phase_total)/df = (1/eta) * dphi_22/d(f*M_s) * d(f*M_s)/df
#                    = (1/eta) * dphi_22/dMf * M_s
# So dphi_22/dMf = eta / M_s * d(Phase_total)/df

print("\n--- 22-mode phase derivative comparison ---")
print(f"{'f(Hz)':>8} {'d(Phi_tot)/df':>16} {'dphi_22/dMf (from LAL)':>22} {'eta/M_s':>12}")
eta_over_Ms = float(eta) / M_s
print(f"  eta/M_s = {eta_over_Ms:.6e}")

for f_check in [20, 50, 100, 200, 500, 1000, 1500, 2000]:
    idx = np.argmin(np.abs(f_arr - f_check))
    if 2 <= idx < len(dphi_total_df) - 2:
        dphi_tot_df = dphi_total_df[idx]
        dphi_22_dMf_from_lal = eta_over_Ms * dphi_tot_df
        print(f"{f_arr[idx]:8.2f} {dphi_tot_df:+16.6e} {dphi_22_dMf_from_lal:+22.6e}")

# Now compute ripple's dphi_raw/dMf at the same frequencies
print("\n--- Ripple raw phase derivative ---")
for f_check in [20, 50, 100, 200, 500, 1000, 1500, 2000]:
    f_val = f_check
    fM_s = f_val * M_s

    # Determine region
    if fM_s < f1_Ms:
        dphi_raw = float(jax.grad(get_inspiral_phase)(fM_s, theta_bbh, bbh_phase_coeffs))
        region = "inspiral"
    elif fM_s < f2_Ms:
        # Intermediate region
        dphi_Ins_match_f1 = float(jax.grad(get_inspiral_phase)(f1_Ms, theta_bbh, bbh_phase_coeffs))
        _, (_, CV_phase_RD0) = get_mergerringdown_raw_phase(f2_Ms, theta_bbh, bbh_phase_coeffs)
        _, (cL, _) = get_mergerringdown_raw_phase(f2_Ms, theta_bbh, bbh_phase_coeffs)
        dphi_Int_match_f1 = float(jax.grad(get_intermediate_raw_phase)(
            f1_Ms, theta_bbh, bbh_phase_coeffs, dphi_Ins_match_f1, CV_phase_RD0, cL
        ))
        alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1

        phi_Int_func = lambda fM_s_: (
            get_intermediate_raw_phase(
                fM_s_, theta_bbh, bbh_phase_coeffs, dphi_Ins_match_f1, CV_phase_RD0, cL
            )
            + alpha1 * fM_s_
        )
        dphi_raw = float(jax.grad(phi_Int_func)(fM_s))
        region = "intermediate"
    else:
        dphi_raw = float(jax.grad(
            lambda x: get_mergerringdown_raw_phase(x, theta_bbh, bbh_phase_coeffs)[0]
        )(fM_s))
        region = "MRD"

    # dphi_22/dMf from ripple (without 1/eta factor)
    # For the intermediate region, we also need the beta1 correction
    if fM_s >= f2_Ms:
        # Need beta1 correction
        phi_Ins_match_f1, dphi_Ins_match_f1_val = jax.value_and_grad(get_inspiral_phase)(
            f1_Ms, theta_bbh, bbh_phase_coeffs
        )
        phi_MRD_match_f2, dphi_MRD_match_f2 = jax.value_and_grad(
            get_mergerringdown_raw_phase, has_aux=True
        )(f2_Ms, theta_bbh, bbh_phase_coeffs)
        _, (cL_val, CV_val) = get_mergerringdown_raw_phase(f2_Ms, theta_bbh, bbh_phase_coeffs)

        phi_Int_match_f1, dphi_Int_match_f1_val = jax.value_and_grad(
            get_intermediate_raw_phase
        )(f1_Ms, theta_bbh, bbh_phase_coeffs, dphi_Ins_match_f1_val, CV_val, cL_val)
        alpha1_val = dphi_Ins_match_f1_val - dphi_Int_match_f1_val

        phi_Int_func = lambda fM_s_: (
            get_intermediate_raw_phase(
                fM_s_, theta_bbh, bbh_phase_coeffs, dphi_Ins_match_f1_val, CV_val, cL_val
            )
            + alpha1_val * fM_s_
        )
        phi_Int_match_f2, dphi_Int_match_f2 = jax.value_and_grad(phi_Int_func)(f2_Ms)
        beta1 = dphi_Int_match_f2 - dphi_MRD_match_f2

        dphi_raw = float(jax.grad(
            lambda x: get_mergerringdown_raw_phase(x, theta_bbh, bbh_phase_coeffs)[0]
        )(fM_s) + beta1)

    print(f"  f={f_val:7.2f} Hz (Mf={fM_s:.8e}, {region}): dphi_raw/dMf = {dphi_raw:+.6e}")

    # Compare with LAL
    idx = np.argmin(np.abs(f_arr - f_check))
    if 2 <= idx < len(dphi_total_df) - 2:
        dphi_22_dMf_from_lal = eta_over_Ms * dphi_total_df[idx]
        print(f"    LAL dphi_22/dMf: {dphi_22_dMf_from_lal:+.6e}")
        print(f"    Difference: {dphi_raw - dphi_22_dMf_from_lal:+.6e}")
        print(f"    Relative: {abs(dphi_raw - dphi_22_dMf_from_lal) / abs(dphi_22_dMf_from_lal):.6e}")
