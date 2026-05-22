#!/usr/bin/env python
"""Debug: check which region f_final falls in and compare PhaseDerivative values."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ripplegw.waveforms.IMRPhenomXAS import (
    Phase as RipplePhase,
    PhaseDerivative as RipplePhaseDeriv,
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

print(f"Matching frequencies (dimensionless M*f):")
print(f"  fMs_MECO (fMs_INmatch): {fMs_MECO:.8f}")
print(f"  fMs_IMmatch: {fMs_IMmatch:.8f}")
print(f"  deltafMs: {deltafMs:.8f}")
print(f"  f1_Ms (inspiral/intermediate boundary): {f1_Ms:.8f}")
print(f"  f2_Ms (intermediate/MRD boundary): {f2_Ms:.8f}")
print(f"  fMs_RD: {fMs_RD:.8f}")
print(f"  fMs_damp: {fMs_damp:.8f}")
print(f"  fMs_ISCO: {fMs_ISCO:.8f}")

# Check f_final
df = 1.0 / 128.0
f_merger = 2037.5291  # approximate
f_final = 2037.5291
f_final_Ms = f_final * M_s

print(f"\nf_final = {f_final:.4f} Hz")
print(f"f_final * M_s = {f_final_Ms:.8f}")
print(f"  In inspiral region? {f_final_Ms < f1_Ms}")
print(f"  In intermediate region? {f1_Ms <= f_final_Ms < f2_Ms}")
print(f"  In MRD region? {f_final_Ms >= f2_Ms}")

# Compute PhaseDerivative at f_final
dphi_deriv = float(RipplePhaseDeriv(f_final, theta_bbh, bbh_phase_coeffs) / M_s)
print(f"\nPhaseDerivative at f_final: {dphi_deriv:.10f}")

# Check with jax.grad(Phase) at f_final
dphi_grad = float(jax.grad(RipplePhase)(f_final, theta_bbh, bbh_phase_coeffs))
print(f"jax.grad(Phase) at f_final: {dphi_grad:.10f}")

# Compute finite difference of Phase at f_final
df_fd = df
dphi_fd = (
    float(RipplePhase(f_final, theta_bbh, bbh_phase_coeffs))
    - float(RipplePhase(f_final - df_fd, theta_bbh, bbh_phase_coeffs))
) / (df_fd * M_s)
print(f"Finite difference of Phase at f_final: {dphi_fd:.10f}")

# Now let's compute the raw phase derivatives directly
print(f"\n--- Raw phase derivatives ---")

# Inspiral
dphi_Ins_match_f1 = float(jax.grad(get_inspiral_phase)(f1_Ms, theta_bbh, bbh_phase_coeffs))
print(f"d(get_inspiral_phase)/dMf at f1_Ms: {dphi_Ins_match_f1:.10f}")

# MRD
_, (cL, CV_phase_RD0) = get_mergerringdown_raw_phase(f2_Ms, theta_bbh, bbh_phase_coeffs)
dphi_MRD_match_f2 = float(
    jax.grad(lambda x: get_mergerringdown_raw_phase(x, theta_bbh, bbh_phase_coeffs)[0])(f2_Ms)
)
print(f"d(get_mergerringdown_raw_phase)/dMf at f2_Ms: {dphi_MRD_match_f2:.10f}")

# Intermediate
dphi_Int_match_f1 = float(
    jax.grad(get_intermediate_raw_phase)(
        f1_Ms, theta_bbh, bbh_phase_coeffs, dphi_Ins_match_f1, CV_phase_RD0, cL
    )
)
print(f"d(get_intermediate_raw_phase)/dMf at f1_Ms: {dphi_Int_match_f1:.10f}")

# Now compute the derivative at f_final_Ms in the MRD region
if f_final_Ms >= f2_Ms:
    dphi_MRD_final = float(
        jax.grad(lambda x: get_mergerringdown_raw_phase(x, theta_bbh, bbh_phase_coeffs)[0])(f_final_Ms)
    )
    print(f"d(get_mergerringdown_raw_phase)/dMf at f_final_Ms: {dphi_MRD_final:.10f}")
    print(f"Divided by eta: {dphi_MRD_final / float(eta):.10f}")
    print(f"Multiplied by M_s: {dphi_MRD_final / float(eta) * M_s:.10f}")
    # Wait, that's not right. The PhaseDerivative already divides by eta and multiplies by M_s.

# Let me trace through PhaseDerivative step by step
print(f"\n--- PhaseDerivative step by step at f_final ---")
fM_s = f_final_Ms

phi_Ins_match_f1, dphi_Ins_match_f1 = jax.value_and_grad(get_inspiral_phase)(
    f1_Ms, theta_bbh, bbh_phase_coeffs
)
phi_MRD_match_f2, dphi_MRD_match_f2 = jax.value_and_grad(
    get_mergerringdown_raw_phase, has_aux=True
)(f2_Ms, theta_bbh, bbh_phase_coeffs)
phi_MRD_match_f2_val, (cL_val, CV_phase_RD0_val) = get_mergerringdown_raw_phase(
    f2_Ms, theta_bbh, bbh_phase_coeffs
)

phi_Int_match_f1, dphi_Int_match_f1 = jax.value_and_grad(
    get_intermediate_raw_phase
)(f1_Ms, theta_bbh, bbh_phase_coeffs, dphi_Ins_match_f1, CV_phase_RD0_val, cL_val)
alpha1 = dphi_Ins_match_f1 - dphi_Int_match_f1
alpha0 = float(phi_Ins_match_f1) - float(phi_Int_match_f1) - float(alpha1) * float(f1_Ms)

phi_Int_func = lambda fM_s_: (
    get_intermediate_raw_phase(
        fM_s_, theta_bbh, bbh_phase_coeffs, dphi_Ins_match_f1, CV_phase_RD0_val, cL_val
    )
    + alpha1 * fM_s_
    + alpha0
)

phi_Int_match_f2, dphi_Int_match_f2 = jax.value_and_grad(phi_Int_func)(f2_Ms)
beta1 = dphi_Int_match_f2 - dphi_MRD_match_f2

dphi_Ins = jax.grad(get_inspiral_phase)(fM_s, theta_bbh, bbh_phase_coeffs)
dphi_Int = jax.grad(phi_Int_func)(fM_s)
dphi_MRD = (
    jax.grad(lambda x: get_mergerringdown_raw_phase(x, theta_bbh, bbh_phase_coeffs)[0])(fM_s)
    + beta1
)

print(f"dphi_Ins (raw): {float(dphi_Ins):.10f}")
print(f"dphi_Int (raw): {float(dphi_Int):.10f}")
print(f"dphi_MRD (raw + beta1): {float(dphi_MRD):.10f}")
print(f"dphi_MRD_raw: {float(jax.grad(lambda x: get_mergerringdown_raw_phase(x, theta_bbh, bbh_phase_coeffs)[0])(fM_s)):.10f}")
print(f"beta1: {float(beta1):.10f}")
print(f"dphi_MRD / eta: {float(dphi_MRD) / float(eta):.10f}")
print(f"dphi_MRD / eta * M_s: {float(dphi_MRD) / float(eta) * M_s:.10f}")

# The PhaseDerivative returns dphase_dMf * M_s where dphase_dMf = dphi_raw / eta
# So the final value should be dphi_MRD / eta * M_s
# But PhaseDerivative is d(Phase)/df, not d(Phase)/dMf
# d(Phase)/df = d(Phase)/dMf * dMf/df = d(Phase)/dMf * M_s
# Phase = (1/eta) * phi_raw, so d(Phase)/dMf = (1/eta) * dphi_raw/dMf
# d(Phase)/df = (1/eta) * dphi_raw/dMf * M_s

# Wait, the raw phase functions take Mf = f * M_s as input.
# So dphi_raw/dMf is already the derivative with respect to Mf.
# And d(Phase)/df = (1/eta) * dphi_raw/dMf * M_s = dphi_raw/dMf / eta * M_s.

# Actually no. The raw phase functions take fM_s (dimensionless frequency) as input.
# Phase(f) = (1/eta) * phi_raw(f * M_s)
# d(Phase)/df = (1/eta) * d(phi_raw)/d(f*M_s) * M_s
#             = (1/eta) * dphi_raw/dMf * M_s

# But wait, that would make d(Phase)/df very large (because M_s is small).
# Let me check: if dphi_raw/dMf ~ -0.0005 and eta ~ 0.25 and M_s ~ 9e-6:
# d(Phase)/df = -0.0005 / 0.25 * 9e-6 = -1.8e-8
# That's way too small!

# Hmm, I think I'm confusing things. Let me re-read the Phase function.
# Phase takes f (Hz) as input, and internally computes fM_s = f * M_s.
# Phase(f) = (1/eta) * phi_raw(f * M_s)
# d(Phase)/df = (1/eta) * d(phi_raw)/d(f*M_s) * d(f*M_s)/df
#             = (1/eta) * dphi_raw/dMf * M_s
# But dphi_raw/dMf has units of rad/(dimensionless freq), and M_s has units of seconds.
# So d(Phase)/df has units of rad/Hz, which is correct.

# But the numerical value... let me think again.
# If phi_raw ~ -50000 at fM_s ~ 0.0005 (f ~ 2000 Hz, M_s ~ 9e-6 s)
# And dphi_raw/dMf ~ -5e8 (very large because phi_raw changes rapidly with Mf)
# Then d(Phase)/df = -5e8 / 0.25 * 9e-6 = -18000
# That's closer to what I'd expect...

# Actually, let me just compute the raw phase derivative at f_final_Ms directly.
print(f"\n--- Direct raw phase derivative at f_final ---")
print(f"f_final_Ms = {f_final_Ms:.8e}")

# MRD raw phase derivative at f_final_Ms
dphi_MRD_direct = jax.grad(lambda x: get_mergerringdown_raw_phase(x, theta_bbh, bbh_phase_coeffs)[0])(fM_s)
print(f"dphi_MRD_raw/dMf = {float(dphi_MRD_direct):.10e}")

# Now with beta1 correction
dphi_MRD_corrected = float(dphi_MRD)
print(f"dphi_MRD_corrected/dMf = {dphi_MRD_corrected:.10e}")

# PhaseDerivative result
phase_deriv_result = float(RipplePhaseDeriv(f_final, theta_bbh, bbh_phase_coeffs))
print(f"PhaseDerivative result (dphase_dMf * M_s) = {phase_deriv_result:.10e}")
print(f"PhaseDerivative / M_s = {phase_deriv_result / M_s:.10e}")

# This should equal d(Phase)/df
# If PhaseDerivative returns dphase_dMf * M_s, then PhaseDerivative / M_s = dphase_dMf
# And d(Phase)/df = dphase_dMf

# Hmm wait, that means PhaseDerivative / M_s = d(Phase)/dMf, not d(Phase)/df
# And d(Phase)/df = d(Phase)/dMf

# No wait. Let me re-read the PhaseDerivative code:
# return dphase_dMf * M_s
# where dphase_dMf = dphi_raw / eta (the derivative of Phase = (1/eta)*phi_raw with respect to Mf)
# So PhaseDerivative returns (dphi_raw/dMf / eta) * M_s

# What is dphi_raw/dMf? It's the derivative of the raw phase with respect to Mf = f*M_s.
# If phi_raw = some_function(f*M_s), then dphi_raw/dMf is dimensionless.
# And PhaseDerivative = (dphi_raw/dMf / eta) * M_s has units of seconds.

# But d(Phase)/df should have units of rad/Hz = seconds.
# Phase = (1/eta) * phi_raw(f*M_s)
# d(Phase)/df = (1/eta) * dphi_raw/dMf * M_s

# So PhaseDerivative = d(Phase)/df. And PhaseDerivative / M_s would be d(Phase)/dMf / M_s... no.
# PhaseDerivative / M_s = dphi_raw/dMf / eta = d(Phase)/dMf

# OK so PhaseDerivative = d(Phase)/df, and the code in NRTidalv3 uses:
# dphiXAS = PhaseDerivative(f_final, ...) / M_s
# This would give d(Phase)/df / M_s = d(Phase)/dMf

# But LAL computes 1/eta * IMRPhenomX_dPhase_22(Mf_final), which is d(Phase)/dMf.
# So ripple's dphiXAS = PhaseDerivative / M_s should equal LAL's 1/eta * dPhase_22.

# But the numerical values don't match! Ripple gives -57, LAL gives 0.003.

# Let me check: what does PhaseDerivative actually return numerically?
print(f"\nPhaseDerivative(f_final) = {phase_deriv_result:.10e}")
print(f"PhaseDerivative(f_final) / M_s = {phase_deriv_result / M_s:.10e}")

# And what does LAL give?
# LAL dphi/df at 2037 Hz = 0.0033 rad/Hz (from the complex value computation)
# If PhaseDerivative = d(Phase)/df, then PhaseDerivative should be ~0.0033
# But it's -57!

# So either PhaseDerivative is wrong, or I'm misunderstanding what it returns.

# Let me compute d(Phase)/df directly from the Phase function:
df_small = 1e-6
phase_plus = float(RipplePhase(f_final + df_small, theta_bbh, bbh_phase_coeffs))
phase_minus = float(RipplePhase(f_final - df_small, theta_bbh, bbh_phase_coeffs))
dPhase_df_direct = (phase_plus - phase_minus) / (2 * df_small)
print(f"\nd(Phase)/df from direct finite difference (df=1e-6 Hz):")
print(f"  Phase(f+df) = {phase_plus:.6e}")
print(f"  Phase(f-df) = {phase_minus:.6e}")
print(f"  d(Phase)/df = {dPhase_df_direct:.10e}")
