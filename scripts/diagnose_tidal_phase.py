#!/usr/bin/env python3
"""Direct phase comparison between ripple and LAL tidal waveform."""
import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.waveforms.IMRPhenomXAS_NRTidalv3 import gen_IMRPhenomXAS_NRTidalv3
from ripplegw.conversions import ms_to_Mc_eta
from ripplegw.constants import MTSUN, PI

m1, m2 = 1.4, 1.4
chi1, chi2 = 0.0, 0.0
lambda1, lambda2 = 1000.0, 1000.0
T = 128.0
df = 1.0 / T
f_ref = 20.0
f_l = 20.0
f_u = 4096.0

M_s = (m1 + m2) * MTSUN
Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))

# Ripple frequency grid (exactly as in test_lal_overlap)
f_sampling = 2 * f_u
delta_t = 1.0 / f_sampling
tlen = int(round(T / delta_t))
freqs_all = np.fft.rfftfreq(tlen, delta_t)
mask = (freqs_all > f_l) & (freqs_all < f_u)
f_arr = freqs_all[mask]
print(f"Frequency grid: {f_arr[0]:.6f} to {f_arr[-1]:.6f} Hz, {len(f_arr)} bins, df={df:.6f} Hz")

# Ripple waveform parameters
dist_mpc = 100.0
tc = 0.0
phi_ref = 0.0  # Use 0 for simple comparison
lambda_tilde = 8.0/13.0 * ((1+7*eta - 31*eta**2)*(lambda1+lambda2) + (1-4*eta)**0.5*(1+9*eta-11*eta**2)*(lambda1-lambda2))
delta_lambda_tilde = 0.5*(1-4*eta)**0.5*(1-(13272./1319.)*eta + (8944./1319.)*eta**2)*(lambda1+lambda2) + (1-(15910./1319.)*eta+(32850./1319.)*eta**2+(3380./1319.)*eta**3)*(lambda1-lambda2)

params = jnp.array([float(Mc), float(eta), chi1, chi2, lambda_tilde, delta_lambda_tilde, dist_mpc, tc, phi_ref])

h_rip = gen_IMRPhenomXAS_NRTidalv3(jnp.array(f_arr), params, f_ref)
h_rip = np.array(h_rip)

print(f"Ripple: {np.sum(np.abs(h_rip) > 0)} nonzero bins")

# LAL waveform
m1_kg, m2_kg = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
dist_SI = dist_mpc * 1e6 * lal.PC_SI

approx = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS_NRTidalv3")
hp_lal, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0, 0, chi1, 0, 0, chi2,
    dist_SI, 0, phi_ref, 0, lambda1, lambda2,
    df, f_l, f_u, f_ref, None, approx,
)
f_arr_lal = np.arange(len(hp_lal.data.data)) * df
h_lal_full = np.array(hp_lal.data.data)

# Extract LAL on ripple grid
h_lal = np.zeros(len(f_arr), dtype=complex)
for i, f in enumerate(f_arr):
    idx = int(round(f / df))
    if 0 <= idx < len(h_lal_full):
        h_lal[i] = h_lal_full[idx]

nz_rip = np.abs(h_rip) > 0
nz_lal = np.abs(h_lal) > 0
nz = nz_rip & nz_lal
print(f"LAL: {np.sum(nz_lal)} nonzero bins on ripple grid")
print(f"Both nonzero: {np.sum(nz)} bins")

# Extract phases
phi_rip = np.angle(h_rip[nz])
phi_lal = np.angle(h_lal[nz])
# Unwrap both
phi_rip_uw = np.unwrap(phi_rip)
phi_lal_uw = np.unwrap(phi_lal)

# Phase difference
delta_phi = phi_rip_uw - phi_lal_uw
f_nz = f_arr[nz]

# Fit linear + constant to phase difference
Mf_nz = f_nz * M_s
A = np.vstack([np.ones_like(Mf_nz), Mf_nz]).T
coeffs, _, _, _ = np.linalg.lstsq(A, delta_phi, rcond=None)
delta_phi_linear = coeffs[0] + coeffs[1] * Mf_nz
delta_phi_residual = delta_phi - delta_phi_linear

print(f"\nPhase difference (ripple - LAL):")
print(f"  Linear fit: constant={coeffs[0]:.6e}, slope={coeffs[1]:.6e} rad/Mf")
print(f"  Residual (nonlinear): max={np.max(np.abs(delta_phi_residual)):.4e} rad")
print(f"  Residual RMS: {np.sqrt(np.mean(delta_phi_residual**2)):.4e} rad")
print(f"  Residual at 20Hz: {float(delta_phi_residual[0]):.4e} rad")
print(f"  Residual at 100Hz: {float(delta_phi_residual[np.argmin(np.abs(f_nz-100))]):.4e} rad")
print(f"  Residual at 500Hz: {float(delta_phi_residual[np.argmin(np.abs(f_nz-500))]):.4e} rad")
print(f"  Residual at f_merger: {float(delta_phi_residual[np.argmin(np.abs(f_nz-1562))]):.4e} rad")

# Compare XAS (BBH only) to isolate the tidal contribution to phase error
from ripplegw.waveforms.IMRPhenomXAS import Phase as XASPhase
from ripplegw.waveforms.IMRPhenomX_utils import PhenomX_phase_coeff_table
theta_bbh = jnp.array([m1, m2, chi1, chi2])

approx_xas = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")
hp_xas, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0, 0, chi1, 0, 0, chi2,
    dist_SI, 0, phi_ref, 0, 0, 0,
    df, f_l, f_u, f_ref, None, approx_xas,
)
h_xas_full = np.array(hp_xas.data.data)
h_xas = np.zeros(len(f_arr), dtype=complex)
for i, f in enumerate(f_arr):
    idx = int(round(f / df))
    if 0 <= idx < len(h_xas_full):
        h_xas[i] = h_xas_full[idx]

nz_xas = np.abs(h_xas) > 0
nz_both_xas = nz_rip & nz_xas

# XAS ripple phase vs LAL XAS phase
phi_xas_rip = np.unwrap(np.angle(np.exp(1j * np.array([float(XASPhase(fi, theta_bbh, PhenomX_phase_coeff_table)) for fi in f_arr[nz_both_xas]]))))
phi_xas_lal = np.unwrap(np.angle(h_xas[nz_both_xas]))
delta_xas = phi_xas_rip - phi_xas_lal
f_xas = f_arr[nz_both_xas]
Mf_xas = f_xas * M_s
A_xas = np.vstack([np.ones_like(Mf_xas), Mf_xas]).T
coeffs_xas, _, _, _ = np.linalg.lstsq(A_xas, delta_xas, rcond=None)
delta_xas_linear = coeffs_xas[0] + coeffs_xas[1] * Mf_xas
delta_xas_residual = delta_xas - delta_xas_linear
print(f"\nXAS Phase difference (ripple - LAL):")
print(f"  Residual max: {np.max(np.abs(delta_xas_residual)):.4e} rad")

# Now isolate tidal contribution to residual
# Phase_tidal_rip(f) = psi_T + psi_QM + psi_SS (from waveform loop)
# Phase_tidal_lal(f) = phaseTidal(f)
# delta_phi_tidal = -(psi_T_rip - phaseTidal_lal)  (tidal enters as subtraction)

# Extract tidal phase from ripple waveform (difference between tidal and BBH)
# h_tidal = h_rip, h_bbh (BBH part of ripple tidal)
# Actually, let's compute bbh part at same alignment
# The phase of h_rip = Phase_bbh + alignment - Phase_tidal_rip
# The phase of h_lal_tidal = Phase_bbh_lal + alignment_lal - Phase_tidal_lal
# delta_phi = (Phase_bbh_rip - Phase_bbh_lal) + (alignment_rip - alignment_lal) + (Phase_tidal_lal - Phase_tidal_rip)
# = delta_xas_aligned + delta_tidal

# So: delta_tidal = delta_phi - delta_xas_aligned (where "aligned" means using the same linear correction)
# But the alignment differs between XAS and tidal tests...

# More directly: compute psi_T + psi_QM + psi_SS from ripple and phaseTidal from LAL
from ripplegw.waveforms.NRTidalv3_utils import (
    get_tidal_phase, get_NRTidalv3_coefficients, get_tidalphasePN_coeffs,
    get_tidal_phase_PN, general_planck_taper, _get_merger_frequency, changePhase_if_min
)
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import get_qm_phase_correction, get_spin_phase_correction

theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
f_merger = float(_get_merger_frequency(theta_intrinsic))
x_arr = PI * jnp.array(f_arr) * M_s
x_23_arr = x_arr ** (2.0/3.0)
f_Ms_arr = jnp.array(f_arr) * M_s
Xa = m1 / (m1 + m2)

PN_coeffs = get_tidalphasePN_coeffs(theta_intrinsic)
NRTidalv3_coeffs = get_NRTidalv3_coefficients(theta_intrinsic, PN_coeffs)
NRTidalv3_phase_arr = np.array(get_tidal_phase(x_arr, NRTidalv3_coeffs, PN_coeffs))
P_P_arr = np.array(general_planck_taper(jnp.array(f_arr), 1.15 * f_merger, 1.35 * f_merger))

# Apply changePhase_if_min
fHzmrgcheck = 0.9 * f_merger
increasing = np.concatenate([[False], NRTidalv3_phase_arr[1:] >= NRTidalv3_phase_arr[:-1]])
valid = (f_arr >= fHzmrgcheck) & increasing
if np.any(valid):
    idx = np.argmax(valid) - 1
    idx = max(idx, 0)
    tidal_min_value = NRTidalv3_phase_arr[idx]
    mask = np.arange(len(f_arr)) >= idx
    NRTidalv3_phase_arr = np.where(mask, tidal_min_value, NRTidalv3_phase_arr)
    print(f"\nchangePhase_if_min activated at f={f_arr[max(idx,0)]:.2f} Hz (idx={idx})")
else:
    print("\nchangePhase_if_min NOT activated")

psi_T_rip = NRTidalv3_phase_arr * (1 - P_P_arr) + np.array(get_tidal_phase_PN(x_arr, Xa, lambda1, lambda2, PN_coeffs)) * P_P_arr
psi_QM_rip = np.array(get_qm_phase_correction(f_Ms_arr, theta_intrinsic))
psi_SS_rip = np.array(get_spin_phase_correction(x_23_arr, theta_intrinsic))
psi_total_rip = psi_T_rip + psi_QM_rip + psi_SS_rip

# Get LAL tidal waveform to extract phaseTidal
# phaseTidal_LAL = arg(h_lal_bbh / h_lal_tidal) + ... (this is hard to extract directly)
# Instead, compare h_lal_tidal to h_lal_bbh on the same grid
h_lal_tidal_full = h_lal_full  # tidal
h_lal_bbh_full = h_xas_full    # BBH (same alignment/distance)

# The phases are: phi_tidal = phi_bbh - phaseTidal; so phaseTidal = phi_bbh - phi_tidal
h_lal_bbh_on_rip = h_xas  # already computed above
nz_joint = nz & nz_both_xas

phi_tidal_lal = np.unwrap(np.angle(h_lal_bbh_on_rip[nz_joint])) - np.unwrap(np.angle(h_lal[nz_joint]))
f_joint = f_arr[nz_joint]
Mf_joint = f_joint * M_s

# Ripple tidal phase on same grid
psi_total_rip_joint = psi_total_rip[nz_joint]

delta_tidal = phi_tidal_lal - psi_total_rip_joint
print(f"\nTidal phase comparison (phi_tidal_lal - psi_total_rip):")
print(f"  f range: {f_joint[0]:.1f} - {f_joint[-1]:.1f} Hz")
A_tidal = np.vstack([np.ones_like(Mf_joint), Mf_joint]).T
coeffs_tidal, _, _, _ = np.linalg.lstsq(A_tidal, delta_tidal, rcond=None)
delta_tidal_linear = coeffs_tidal[0] + coeffs_tidal[1] * Mf_joint
delta_tidal_residual = delta_tidal - delta_tidal_linear
print(f"  Linear fit: const={coeffs_tidal[0]:.6e}, slope={coeffs_tidal[1]:.6e}")
print(f"  Residual max: {np.max(np.abs(delta_tidal_residual)):.4e} rad")
print(f"  Residual RMS: {np.sqrt(np.mean(delta_tidal_residual**2)):.4e} rad")

# Check at key frequencies
for fcheck in [20, 50, 100, 200, 500, 1000, 1500]:
    idx_c = np.argmin(np.abs(f_joint - fcheck))
    print(f"  f={fcheck} Hz: delta_phi={delta_tidal[idx_c]:.6e}, residual={delta_tidal_residual[idx_c]:.6e}")
