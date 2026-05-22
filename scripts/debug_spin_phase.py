#!/usr/bin/env python
"""Debug spin phase correction discrepancy between LAL and Ripple."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, MPC, PI, TWO_PI, MRSUN
from ripplegw.waveforms.IMRPhenom_tidal_utils import get_quadparam_octparam

jax.config.update("jax_enable_x64", True)


def compute_spin_phase_lal_direct(f, m1, m2, chi1, chi2, lambda1, lambda2):
    """Compute spin phase correction exactly as LAL does it, step by step."""
    M = m1 + m2
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    piM = PI * M_s
    
    X_A = m1 / M
    X_B = m2 / M
    
    # LAL computes dQuadMon from lambda using universal relation
    def get_quadparam_from_lambda(lam):
        """This is what LAL's SimUniversalRelationQuadMonVSlambda2Tidal does."""
        if lam < 1:
            quadparam = 1.0 + lam * (
                0.427688866723244
                + lam * (-0.324336526985068 + lam * 0.1107439432180572)
            )
            return quadparam
        else:
            quad_coeffs = np.array([0.1940, 0.09163, 0.04812, -4.283e-3, 1.245e-4])
            log_lambda = np.log(lam)
            log_quadparam = (
                quad_coeffs[0]
                + quad_coeffs[1] * log_lambda
                + quad_coeffs[2] * log_lambda**2
                + quad_coeffs[3] * log_lambda**3
                + quad_coeffs[4] * log_lambda**4
            )
            return np.exp(log_quadparam)
    
    quadparam1 = get_quadparam_from_lambda(lambda1)
    quadparam2 = get_quadparam_from_lambda(lambda2)
    
    # Octupole from quadrupole (LAL's XLALSimUniversalRelationSpinInducedOctupoleVSSpinInducedQuadrupole)
    def get_octparam_from_quadparam(quadparam):
        oct_coeffs = np.array([0.003131, 2.071, -0.7152, 0.2458, -0.03309])
        log_quadparam = np.log(quadparam)
        log_octparam = (
            oct_coeffs[0]
            + oct_coeffs[1] * log_quadparam
            + oct_coeffs[2] * log_quadparam**2
            + oct_coeffs[3] * log_quadparam**3
            + oct_coeffs[4] * log_quadparam**4
        )
        return np.exp(log_octparam)
    
    octparam1 = get_octparam_from_quadparam(quadparam1)
    octparam2 = get_octparam_from_quadparam(quadparam2)
    
    print(f"  quadparam1 = {quadparam1:.15e}")
    print(f"  quadparam2 = {quadparam2:.15e}")
    print(f"  octparam1 = {octparam1:.15e}")
    print(f"  octparam2 = {octparam2:.15e}")
    
    # LAL's XLALSimInspiralGetHOSpinTerms
    # Called with: (..., X_A, X_B, chi1, chi2, dquadmon1+1, dquadmon2+1)
    # where dquadmon1 = quadparam1 - 1, so dquadmon1 + 1 = quadparam1
    # Inside the function, it does (quadparam1 - 1) and (octparam1 - 1)
    
    chi1_sq = chi1**2
    chi2_sq = chi2**2
    X_Asq = X_A**2
    X_Bsq = X_B**2
    
    # SS_3p5PN
    SS_3p5PN = (-400.0 * PI * (quadparam1 - 1.0) * chi1_sq * X_Asq
                - 400.0 * PI * (quadparam2 - 1.0) * chi2_sq * X_Bsq)
    
    # SSS_3p5PN
    SSS_3p5PN = (
        10.0 * ((X_Asq + 308.0 / 3.0 * X_A) * chi1 + (X_Bsq - 89.0 / 3.0 * X_B) * chi2)
        * (quadparam1 - 1.0) * X_Asq * chi1_sq
        + 10.0 * ((X_Bsq + 308.0 / 3.0 * X_B) * chi2 + (X_Asq - 89.0 / 3.0 * X_A) * chi1)
        * (quadparam2 - 1.0) * X_Bsq * chi2_sq
        - 440.0 * (octparam1 - 1.0) * X_A * X_Asq * chi1_sq * chi1
        - 440.0 * (octparam2 - 1.0) * X_B * X_Bsq * chi2_sq * chi2
    )
    
    print(f"  SS_3p5PN = {SS_3p5PN:.15e}")
    print(f"  SSS_3p5PN = {SSS_3p5PN:.15e}")
    print(f"  SS + SSS = {SS_3p5PN + SSS_3p5PN:.15e}")
    
    # pn_fac
    pn_fac = 3.0 * piM ** (2.0 / 3.0) / (128.0 * eta)
    print(f"  pn_fac = {pn_fac:.15e}")
    
    # Final spin phase: pn_fac * (SS_3p5PN + SSS_3p5PN) * f^(2/3)
    spin_phase = pn_fac * (SS_3p5PN + SSS_3p5PN) * f ** (2.0 / 3.0)
    
    return spin_phase, SS_3p5PN, SSS_3p5PN


def compute_spin_phase_ripple(f, m1, m2, chi1, chi2, lambda1, lambda2):
    """Compute spin phase correction using Ripple's implementation."""
    from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import get_spin_phase_correction
    
    M = m1 + m2
    M_s = M * MTSUN
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    
    x_val = jnp.array([(PI * M_s * f) ** (2.0 / 3.0)])
    psi_SS = float(get_spin_phase_correction(x_val, theta_intrinsic)[0])
    
    return psi_SS


def main():
    m1, m2 = 1.4, 1.3
    chi1, chi2 = 0.02, -0.01
    lambda1, lambda2 = 400.0, 300.0
    
    print(f"=== Spin phase correction deep debug ===")
    print(f"m1={m1}, m2={m2}, chi1={chi1}, chi2={chi2}")
    print(f"lambda1={lambda1}, lambda2={lambda2}")
    print()
    
    # Compare at a few frequencies
    test_freqs = [50.0, 100.0, 500.0, 1000.0, 2000.0]
    
    for f in test_freqs:
        print(f"f = {f} Hz:")
        
        print("  LAL computation:")
        spin_lal, SS_lal, SSS_lal = compute_spin_phase_lal_direct(f, m1, m2, chi1, chi2, lambda1, lambda2)
        print(f"  LAL spin_phase = {spin_lal:.15e}")
        
        print("  Ripple computation:")
        spin_ripple = compute_spin_phase_ripple(f, m1, m2, chi1, chi2, lambda1, lambda2)
        print(f"  Ripple spin_phase = {spin_ripple:.15e}")
        
        print(f"  Difference = {spin_ripple - spin_lal:.15e}")
        print()
    
    # Now let's look at what Ripple's get_spin_phase_correction does step by step
    print(f"=== Ripple's get_spin_phase_correction internals ===")
    
    M = m1 + m2
    m1_s = m1 * MTSUN
    m2_s = m2 * MTSUN
    M_s = m1_s + m2_s
    eta = m1_s * m2_s / (M_s**2.0)
    X1 = m1_s / M_s
    X2 = m2_s / M_s
    X1sq = X1 * X1
    X2sq = X2 * X2
    chi1_sq = chi1 * chi1
    chi2_sq = chi2 * chi2
    
    print(f"  eta = {eta:.15e}")
    print(f"  X1 = {X1:.15e}, X2 = {X2:.15e}")
    print(f"  X1sq = {X1sq:.15e}, X2sq = {X2sq:.15e}")
    print(f"  chi1_sq = {chi1_sq:.15e}, chi2_sq = {chi2_sq:.15e}")
    
    # Get quadparam and octparam using Ripple's function
    quadparam1_r, octparam1_r = get_quadparam_octparam(jnp.array(lambda1))
    quadparam2_r, octparam2_r = get_quadparam_octparam(jnp.array(lambda2))
    quadparam1_r = float(quadparam1_r) - 1
    quadparam2_r = float(quadparam2_r) - 1
    octparam1_r = float(octparam1_r) - 1
    octparam2_r = float(octparam2_r) - 1
    
    print(f"  quadparam1-1 = {quadparam1_r:.15e}")
    print(f"  quadparam2-1 = {quadparam2_r:.15e}")
    print(f"  octparam1-1 = {octparam1_r:.15e}")
    print(f"  octparam2-1 = {octparam2_r:.15e}")
    
    # SS_2
    SS_2 = (-50.0 * quadparam1_r * chi1_sq * X1sq 
            - 50.0 * quadparam2_r * chi2_sq * X2sq)
    print(f"  SS_2 = {SS_2:.15e}")
    
    # SS_3
    SS_3 = (5.0 / 84.0 * (9407.0 + 8218.0 * X1 - 2016.0 * X1sq) * quadparam1_r * X1sq * chi1_sq
            + 5.0 / 84.0 * (9407.0 + 8218.0 * X2 - 2016.0 * X2sq) * quadparam2_r * X2sq * chi2_sq)
    print(f"  SS_3 = {SS_3:.15e}")
    
    # SS_3p5
    SS_3p5 = (-400.0 * PI * quadparam1_r * chi1_sq * X1sq 
              - 400.0 * PI * quadparam2_r * chi2_sq * X2sq)
    print(f"  SS_3p5 = {SS_3p5:.15e}")
    
    # SSS_3p5
    SSS_3p5 = (
        10.0 * ((X1sq + 308.0 / 3.0 * X1) * chi1 + (X2sq - 89.0 / 3.0 * X2) * chi2)
        * quadparam1_r * X1sq * chi1_sq
        + 10.0 * ((X2sq + 308.0 / 3.0 * X2) * chi2 + (X1sq - 89.0 / 3.0 * X1) * chi1)
        * quadparam2_r * X2sq * chi2_sq
        - 440.0 * octparam1_r * X1 * X1sq * chi1_sq * chi1
        - 440.0 * octparam2_r * X2 * X2sq * chi2_sq * chi2
    )
    print(f"  SSS_3p5 = {SSS_3p5:.15e}")
    
    # prefac
    prefac = 3.0 / (128.0 * eta)
    print(f"  prefac = {prefac:.15e}")
    
    # At f=50 Hz
    f = 50.0
    M_s = M * MTSUN
    piM = PI * M_s
    x = (piM * f) ** (2.0 / 3.0)
    
    print(f"\n  At f={f} Hz:")
    print(f"    x = (piM*f)^(2/3) = {x:.15e}")
    print(f"    x^(-1/2) = {x**(-0.5):.15e}")
    print(f"    x^(1/2) = {x**0.5:.15e}")
    print(f"    x = {x:.15e}")
    
    # Full phase
    psi_SS = prefac * (SS_2 * x ** (-0.5) + SS_3 * x ** 0.5 + (SS_3p5 + SSS_3p5) * x)
    print(f"    psi_SS = {psi_SS:.15e}")
    
    # Compare with LAL
    print(f"\n  LAL spin_phase at f={f}:")
    spin_lal, SS_lal, SSS_lal = compute_spin_phase_lal_direct(f, m1, m2, chi1, chi2, lambda1, lambda2)
    print(f"    LAL = {spin_lal:.15e}")
    print(f"    Ripple = {psi_SS:.15e}")
    print(f"    Diff = {psi_SS - spin_lal:.15e}")
    
    print("\n=== Key insight ===")
    print("LAL's spin phase correction:")
    print("  psi_spin = pn_fac * (SS_3p5PN + SSS_3p5PN) * f^(2/3)")
    print("  where pn_fac = 3 * piM^(2/3) / (128 * eta)")
    print("  and f^(2/3) is computed with f in Hz")
    print()
    print("Ripple's spin phase correction:")
    print("  psi_SS = prefac * (SS_2 * x^(-1/2) + SS_3 * x^(1/2) + (SS_3p5 + SSS_3p5) * x)")
    print("  where prefac = 3 / (128 * eta)")
    print("  and x = (piM * f)^(2/3)")
    print()
    print("Let's expand Ripple's formula for the 3.5PN term:")
    print("  prefac * (SS_3p5 + SSS_3p5) * x")
    print("  = 3/(128*eta) * (SS_3p5 + SSS_3p5) * (piM*f)^(2/3)")
    print("  = 3*(piM)^(2/3)/(128*eta) * (SS_3p5 + SSS_3p5) * f^(2/3)")
    print("  = pn_fac * (SS_3p5 + SSS_3p5) * f^(2/3)")
    print()
    print("This is IDENTICAL to LAL's formula for the 3.5PN term!")
    print("So the 3.5PN term should match...")
    print()
    print("But Ripple ALSO has SS_2 and SS_3 terms (2PN and 3PN)!")
    print("LAL does NOT have these terms in the waveform assembly.")
    print()
    print("BUG FOUND: Ripple includes SS_2 (2PN) and SS_3 (3PN) spin terms")
    print("that LAL does NOT apply in IMRPhenomD_NRTidalv2!")
    
    # Verify this by computing just the 3.5PN part
    print(f"\n=== Verification: 3.5PN term only ===")
    psi_3p5_only = prefac * (SS_3p5 + SSS_3p5) * x
    print(f"  Ripple 3.5PN only: {psi_3p5_only:.15e}")
    print(f"  LAL full:          {spin_lal:.15e}")
    print(f"  Match: {abs(psi_3p5_only - spin_lal) < 1e-15}")


if __name__ == "__main__":
    main()
