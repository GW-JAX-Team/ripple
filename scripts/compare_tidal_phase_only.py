#!/usr/bin/env python3
"""Compare ripple's tidal phase computation against LAL's NRTidal functions directly."""
import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.waveforms.NRTidalv3_utils import (
    _get_merger_frequency, phenomx_tidal_phase,
    get_tidal_phase, get_NRTidalv3_coefficients, get_tidalphasePN_coeffs,
    get_tidal_phase_PN, general_planck_taper, changePhase_if_min
)
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import get_qm_phase_correction, get_spin_phase_correction
from ripplegw.constants import MTSUN, PI

m1, m2 = 1.4, 1.4
chi1, chi2 = 0.0, 0.0
lambda1, lambda2 = 1000.0, 1000.0
T = 128.0
df = 1.0 / T
f_l = 20.0
f_u = 4096.0

M_s = (m1 + m2) * MTSUN
theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
Xa = m1 / (m1 + m2)

f_merger = float(_get_merger_frequency(theta_intrinsic))
print(f"f_merger = {f_merger:.4f} Hz, Mf_merger = {f_merger * M_s:.8f}")

# Build frequency array (same as ripple's test)
f_sampling = 2 * f_u
delta_t = 1.0 / f_sampling
tlen = int(round(T / delta_t))
freqs_all = np.fft.rfftfreq(tlen, delta_t)
mask = (freqs_all > f_l) & (freqs_all < f_u)
f_arr = freqs_all[mask]
Mf_arr = f_arr * M_s
x_arr = PI * Mf_arr
x23_arr = x_arr**(2.0/3.0)
f_Ms_arr = jnp.array(Mf_arr)

# Compute ripple tidal phase at each frequency
PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)
NRTidalv3_phase = np.array(get_tidal_phase(jnp.array(x_arr), NRTidalv3_coeffs, PN_coeffs))

P_P = np.array(general_planck_taper(jnp.array(f_arr), 1.15 * f_merger, 1.35 * f_merger))

# Apply changePhase_if_min
fHzmrgcheck = 0.9 * f_merger
NRT_mod = NRTidalv3_phase.copy()
increasing = np.concatenate([[False], NRT_mod[1:] >= NRT_mod[:-1]])
valid = (f_arr >= fHzmrgcheck) & increasing
if np.any(valid):
    idx = np.argmax(valid) - 1
    idx = max(idx, 0)
    tidal_min = NRT_mod[idx]
    msk = np.arange(len(f_arr)) >= idx
    NRT_mod = np.where(msk, tidal_min, NRT_mod)
    print(f"changePhase_if_min activated at f={f_arr[idx]:.2f} Hz")
else:
    print("changePhase_if_min NOT activated")

psi_T = NRT_mod * (1 - P_P) + np.array(get_tidal_phase_PN(jnp.array(x_arr), Xa, lambda1, lambda2, PN_coeffs)) * P_P
psi_QM = np.array(get_qm_phase_correction(f_Ms_arr, theta_intrinsic))
psi_SS = np.array(get_spin_phase_correction(jnp.array(x23_arr), theta_intrinsic))
psi_rip = psi_T + psi_QM + psi_SS

# Get LAL's tidal phase directly via XLALSimNRTunedTidesFDTidalPhaseFrequencySeries
m1_kg, m2_kg = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
Lambda1_lal = lambda1
Lambda2_lal = lambda2
chi1_lal = chi1
chi2_lal = chi2

# Use LAL's function to compute just the tidal phase frequency series
# We need LAL's SimNRTunedTidesFDTidalPhaseFrequencySeries_v3
# For the spin-tidal terms we use LAL's IMRPhenomX_TidalPhase equivalent

# Let's compare by extracting from the actual waveform
# LAL NRTidalv3 phase = phi_BBH - phi_tidal_waveform
# We need to get the tidal phase + alignment separately

# Alternative: directly call LAL tidal functions
try:
    # Try to call XLALSimNRTunedTidesFDTidalPhase_v3 directly
    psi_T_lal_arr = np.zeros(len(f_arr))
    for i, f in enumerate(f_arr[:10]):
        Mf = f * M_s
        M_omega = PI * Mf
        lal_phase = lalsim.SimNRTunedTidesFDTidalPhase_v3(M_omega, m1_kg, m2_kg, Lambda1_lal, Lambda2_lal)
        psi_T_lal_arr[i] = lal_phase
    print(f"\nDirect LAL tidal phase at f={f_arr[0]:.4f}: {psi_T_lal_arr[0]:.8e}")
    print(f"Ripple NRTidalv3_phase at f={f_arr[0]:.4f}: {NRTidalv3_phase[0]:.8e}")
    print(f"Diff: {psi_T_lal_arr[0] - NRTidalv3_phase[0]:.4e}")
except Exception as e:
    print(f"\nCannot call SimNRTunedTidesFDTidalPhase_v3 directly: {e}")

# Compare phenomx_tidal_phase to what we compute in the waveform loop
print(f"\nComparing phenomx_tidal_phase vs waveform-loop tidal phase:")
check_freqs = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 1400.0, 1562.0]
print(f"{'f (Hz)':>10} {'phenomx':>14} {'waveform_loop':>14} {'diff':>12}")
for fc in check_freqs:
    idx = np.argmin(np.abs(f_arr - fc))
    psi_loop = float(psi_rip[idx])
    psi_func = float(phenomx_tidal_phase(theta_intrinsic, f_arr[idx] * M_s))
    diff = psi_loop - psi_func
    print(f"{f_arr[idx]:>10.2f} {psi_func:>14.6e} {psi_loop:>14.6e} {diff:>12.4e}")

# Also verify get_tidal_phase vs phenomx_tidal_phase NR part
print(f"\nVerifying get_tidal_phase vs phenomx_tidal_phase NR part:")
for fc in [20.0, 100.0, 500.0, 1000.0]:
    idx = np.argmin(np.abs(f_arr - fc))
    Mf_c = f_arr[idx] * M_s
    x_c = PI * Mf_c

    psi_phx = float(phenomx_tidal_phase(theta_intrinsic, Mf_c))  # includes spin terms
    psi_nrt = float(get_tidal_phase(jnp.array([x_c]), NRTidalv3_coeffs, PN_coeffs)[0])
    psi_pn = float(get_tidal_phase_PN(jnp.array([x_c]), Xa, lambda1, lambda2, PN_coeffs)[0])
    P_P_c = float(general_planck_taper(jnp.array(Mf_c), 1.15 * f_merger * M_s, 1.35 * f_merger * M_s))

    # phenomx_tidal_phase uses: NR*(1-P_P) + PN*P_P (but computes in Mf units)
    # waveform loop uses: NRTidalv3_phase*(1-P_P_f) + PN_phase*P_P_f (in f units)
    # These should be equal since Mf/Mf_merger = f/f_merger
    P_P_f = float(general_planck_taper(jnp.array(f_arr[idx]), 1.15 * f_merger, 1.35 * f_merger))
    psi_combined = psi_nrt * (1-P_P_f) + psi_pn * P_P_f
    print(f"  f={fc:.0f} Hz: P_P_Mf={P_P_c:.6f}, P_P_f={P_P_f:.6f}, diff_PP={P_P_c-P_P_f:.2e}")
    print(f"           NR={psi_nrt:.6e}, PN={psi_pn:.6e}, combined={psi_combined:.6e}, phenomx_full={psi_phx:.6e}")
