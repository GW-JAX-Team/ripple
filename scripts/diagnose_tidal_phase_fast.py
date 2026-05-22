#!/usr/bin/env python3
"""Fast tidal phase comparison: focus on the alignment formulas."""
import jax
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim

jax.config.update("jax_enable_x64", True)

from ripplegw.waveforms.IMRPhenomXAS import Phase as XASPhase, PhaseDerivative
from ripplegw.waveforms.IMRPhenomX_utils import PhenomX_phase_coeff_table, calc_phaseatpeak, get_cutoff_fMs
from ripplegw.waveforms.NRTidalv3_utils import (
    _get_merger_frequency, phenomx_tidal_phase, phenomx_tidal_phase_derivative,
    get_tidal_phase, get_NRTidalv3_coefficients, get_tidalphasePN_coeffs,
    get_tidal_phase_PN, general_planck_taper, changePhase_if_min
)
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import get_qm_phase_correction, get_spin_phase_correction
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
theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
theta_bbh = jnp.array([m1, m2, chi1, chi2])
bbh_phase_coeffs = PhenomX_phase_coeff_table

f_merger = float(_get_merger_frequency(theta_intrinsic))
print(f"f_merger = {f_merger:.4f} Hz, Mf_merger = {f_merger * M_s:.6f}")

# Compute f_final as ripple does
f_sampling = 2 * f_u
delta_t = 1.0 / f_sampling
tlen = int(round(T / delta_t))
freqs_all = np.fft.rfftfreq(tlen, delta_t)
mask = (freqs_all > f_l) & (freqs_all < f_u)
f_arr = freqs_all[mask]
f_final = min(f_arr[-1] + df, f_merger)
print(f"f_final = {f_final:.6f} Hz")

# Ripple linb computation
delta = float(jnp.sqrt(1.0 - 4.0 * float(eta)))
mm1 = 0.5 * (1.0 + delta)
mm2 = 0.5 * (1.0 - delta)
StotR = (mm1**2 + mm2**2)**(-1.0) * (mm1**2 * chi1 + mm2**2 * chi2)
chia = chi1 - chi2
fMs_RD, fMs_damp, _, _ = get_cutoff_fMs(m1, m2, chi1, chi2)
lina, linb_init, psi4tostrain = calc_phaseatpeak(float(eta), StotR, chia, delta)
dphi22Ref = float(jax.grad(XASPhase)((fMs_RD - fMs_damp) / M_s, theta_bbh, bbh_phase_coeffs) / M_s)
linb_BBH = float(linb_init) - dphi22Ref - 2.0 * PI * (500.0 + float(psi4tostrain))

dphiXAS = (float(XASPhase(f_final, theta_bbh, bbh_phase_coeffs)) -
           float(XASPhase(f_final - df, theta_bbh, bbh_phase_coeffs))) / (df * M_s)
dphiT = float(phenomx_tidal_phase_derivative(theta_intrinsic, f_final * M_s))
linb_rip = dphiT - dphiXAS
phiTfRef_rip = float(phenomx_tidal_phase(theta_intrinsic, f_ref * M_s))
phifRef_rip = -(float(XASPhase(f_ref, theta_bbh, bbh_phase_coeffs)) + linb_rip * (f_ref * M_s) + float(lina) - phiTfRef_rip) + PI/4 + PI

print(f"\nRipple alignment:")
print(f"  linb_BBH = {linb_BBH:.8e}")
print(f"  dphiXAS = {dphiXAS:.8e}")
print(f"  dphiT = {dphiT:.8e}")
print(f"  linb = {linb_rip:.8e}")
print(f"  phiTfRef = {phiTfRef_rip:.8e}")
print(f"  phifRef = {phifRef_rip:.8e}")

# Get LAL alignment via direct IMRPhenomXAS_NRTidalv3 waveform
m1_kg, m2_kg = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
dist_SI = 100e6 * lal.PC_SI
phi_ref_lal = 0.0

approx_tidal = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS_NRTidalv3")
hp_tidal, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0, 0, chi1, 0, 0, chi2,
    dist_SI, 0, phi_ref_lal, 0, lambda1, lambda2,
    df, f_l, f_u, f_ref, None, approx_tidal,
)
f_arr_lal = np.arange(len(hp_tidal.data.data)) * df
h_tidal_lal = np.array(hp_tidal.data.data)

approx_xas = lalsim.SimInspiralGetApproximantFromString("IMRPhenomXAS")
hp_xas, _ = lalsim.SimInspiralChooseFDWaveform(
    m1_kg, m2_kg, 0, 0, chi1, 0, 0, chi2,
    dist_SI, 0, phi_ref_lal, 0, 0, 0,
    df, f_l, f_u, f_ref, None, approx_xas,
)
h_xas_lal = np.array(hp_xas.data.data)

# Extract LAL tidal phase = phi_bbh_lal - phi_tidal_lal
nz_tidal = np.where(np.abs(h_tidal_lal) > 0)[0]
nz_xas = np.where(np.abs(h_xas_lal) > 0)[0]
print(f"\nLAL tidal: {f_arr_lal[nz_tidal[0]]:.4f} to {f_arr_lal[nz_tidal[-1]]:.4f} Hz")

# Extract tidal phase at f_ref (use phi_bbh - phi_tidal)
idx_ref = int(round(f_ref / df))
if np.abs(h_tidal_lal[idx_ref]) > 0 and np.abs(h_xas_lal[idx_ref]) > 0:
    phi_tidal_lal_ref = np.angle(h_xas_lal[idx_ref]) - np.angle(h_tidal_lal[idx_ref])
    print(f"phi_tidal_lal at f_ref: {phi_tidal_lal_ref:.8e}")
else:
    print(f"LAL waveform zero at f_ref!")

# Compute tidal phase derivative at f_final from LAL (via finite diff of the tidal waveform)
idx_final = int(round(f_final / df))
idx_prev = idx_final - 1
if idx_prev >= 0 and np.abs(h_tidal_lal[idx_final]) > 0 and np.abs(h_tidal_lal[idx_prev]) > 0:
    # phi_tidal(f) = phi_bbh(f) - phi_waveform(f)
    # d(phi_tidal)/df ≈ (phi_tidal(f) - phi_tidal(f-df)) / df
    def extract_tidal_phase_at(idx):
        if np.abs(h_xas_lal[idx]) > 0 and np.abs(h_tidal_lal[idx]) > 0:
            return np.angle(h_xas_lal[idx]) - np.angle(h_tidal_lal[idx])
        return None

    phi_T_final = np.unwrap([extract_tidal_phase_at(idx_prev), extract_tidal_phase_at(idx_final)])
    dphiT_lal = (phi_T_final[1] - phi_T_final[0]) / (df * M_s)
    print(f"\nLAL dphiT at f_final (finite diff): {dphiT_lal:.8e}")
    print(f"Rip dphiT at f_final (analytic):     {dphiT:.8e}")
    print(f"Difference: {dphiT - dphiT_lal:.4e}")
else:
    print(f"Cannot extract LAL dphiT at f_final")

# Compare phiTfRef between ripple and LAL
print(f"\nphiTfRef comparison:")
print(f"  Ripple phiTfRef = {phiTfRef_rip:.8e}")
if np.abs(h_tidal_lal[idx_ref]) > 0 and np.abs(h_xas_lal[idx_ref]) > 0:
    phi_tidal_lal_at_ref = np.angle(h_xas_lal[idx_ref]) - np.angle(h_tidal_lal[idx_ref])
    print(f"  LAL phi_tidal at f_ref (angle diff): {phi_tidal_lal_at_ref:.8e}")
    print(f"  Note: This includes alignment offsets; not directly comparable")

# Direct comparison: compute total tidal phase at several frequencies
print(f"\n--- Direct tidal phase comparison at specific frequencies ---")
check_freqs = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 1400.0]
print(f"{'f (Hz)':>10} {'psi_rip':>16} {'phi_lal_diff':>16} {'difference':>14}")

for fc in check_freqs:
    idx_c = int(round(fc / df))
    if idx_c >= 1 and np.abs(h_tidal_lal[idx_c]) > 0 and np.abs(h_xas_lal[idx_c]) > 0:
        # Get ripple tidal phase at fc
        Mfc = fc * M_s
        x_c = PI * Mfc
        x23_c = x_c**(2.0/3.0)
        psi_T_c = float(phenomx_tidal_phase(theta_intrinsic, Mfc))  # includes all tidal terms

        # Get LAL tidal phase at fc (difference between BBH and tidal waveform phases)
        # Use unwrapped phase from a range
        # Use two points for unwrapping
        if idx_c >= 2 and np.abs(h_tidal_lal[idx_c-1]) > 0 and np.abs(h_xas_lal[idx_c-1]) > 0:
            phi_t_prev = np.angle(h_xas_lal[idx_c-1]) - np.angle(h_tidal_lal[idx_c-1])
            phi_t_curr = np.angle(h_xas_lal[idx_c]) - np.angle(h_tidal_lal[idx_c])
            phi_t_unwrapped = np.unwrap([phi_t_prev, phi_t_curr])
            phi_tidal_lal_fc = phi_t_unwrapped[1]
        else:
            phi_tidal_lal_fc = np.angle(h_xas_lal[idx_c]) - np.angle(h_tidal_lal[idx_c])

        diff = psi_T_c - phi_tidal_lal_fc
        print(f"{fc:>10.1f} {psi_T_c:>16.6e} {phi_tidal_lal_fc:>16.6e} {diff:>14.4e}")
