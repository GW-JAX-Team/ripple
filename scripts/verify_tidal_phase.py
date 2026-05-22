#!/usr/bin/env python
"""Verify tidal phase computation step by step."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, PI
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import get_tidal_phase, get_kappa

jax.config.update("jax_enable_x64", True)


def tidal_phase_lal_style(f, m1, m2, kappa):
    """Compute tidal phase exactly as LAL does it."""
    M = m1 + m2
    Xa = m1 / M
    Xb = m2 / M
    
    # LAL: M_omega = LAL_PI * fHz * (mtot * LAL_MTSUN_SI)
    M_omega = PI * f * (M * MTSUN)
    
    # PN_x = pow(M_omega, 2.0/3.0)
    PN_x = M_omega ** (2.0 / 3.0)
    PN_x_2 = PN_x * PN_x
    PN_x_3 = PN_x * PN_x_2
    PN_x_3over2 = PN_x ** (3.0 / 2.0)
    PN_x_5over2 = PN_x ** (5.0 / 2.0)
    
    # Coefficients
    c_Newt = 2.4375
    n_1 = -12.615214237993088
    n_3over2 = 19.0537346970349
    n_2 = -21.166863146081035
    n_5over2 = 90.55082156324926
    n_3 = -60.25357801943598
    d_1 = -15.111207827736678
    d_3over2 = 22.195327350624694
    d_2 = 8.064109635305156
    
    tidal_phase = -kappa * c_Newt / (Xa * Xb) * PN_x_5over2
    num = 1.0 + n_1 * PN_x + n_3over2 * PN_x_3over2 + n_2 * PN_x_2 + n_5over2 * PN_x_5over2 + n_3 * PN_x_3
    den = 1.0 + d_1 * PN_x + d_3over2 * PN_x_3over2 + d_2 * PN_x_2
    ratio = num / den
    tidal_phase *= ratio
    
    return tidal_phase


def tidal_phase_ripple_style(f, m1, m2, kappa):
    """Compute tidal phase using Ripple's function."""
    M = m1 + m2
    M_s = M * MTSUN
    theta_intrinsic = jnp.array([m1, m2, 0.0, 0.0, 0.0, 0.0])  # chi, lambda don't matter for x
    
    x_val = jnp.array([(PI * M_s * f) ** (2.0 / 3.0)])
    # We need to pass actual theta for get_tidal_phase
    theta = jnp.array([m1, m2, 0.0, 0.0, 0.0, 0.0])
    psi_T = float(get_tidal_phase(x_val, theta, kappa)[0])
    
    return psi_T


def main():
    m1, m2 = 1.4, 1.3
    lambda1, lambda2 = 400.0, 300.0
    
    M = m1 + m2
    X1 = m1 / M
    X2 = m2 / M
    term1 = (1.0 + 12.0 * X2 / X1) * (X1**5.0) * lambda1
    term2 = (1.0 + 12.0 * X1 / X2) * (X2**5.0) * lambda2
    kappa = (3.0 / 13.0) * (term1 + term2)
    
    print(f"kappa = {kappa:.15e}")
    print(f"X1 = {X1:.15e}, X2 = {X2:.15e}")
    print()
    
    test_freqs = [50.0, 100.0, 500.0, 1000.0, 2000.0]
    
    print(f"=== Step-by-step tidal phase comparison ===")
    for f in test_freqs:
        M_s = M * MTSUN
        
        # LAL style
        M_omega = PI * f * M_s
        PN_x = M_omega ** (2.0 / 3.0)
        
        # Ripple style
        x_ripple = (PI * M_s * f) ** (2.0 / 3.0)
        
        print(f"f = {f} Hz:")
        print(f"  M_omega (LAL) = {M_omega:.15e}")
        print(f"  PN_x (LAL) = {PN_x:.15e}")
        print(f"  x (Ripple) = {float(x_ripple):.15e}")
        print(f"  PN_x - x = {PN_x - float(x_ripple):.2e}")
        
        psi_lal = tidal_phase_lal_style(f, m1, m2, kappa)
        psi_rip = tidal_phase_ripple_style(f, m1, m2, kappa)
        print(f"  psi_T (LAL) = {psi_lal:.15e}")
        print(f"  psi_T (Ripple) = {psi_rip:.15e}")
        print(f"  diff = {psi_lal - psi_rip:.2e}")
        print()
    
    # The tidal phases match exactly! So the issue is NOT in the tidal phase computation.
    # The phase difference between LAL and Ripple NRTidalv2 must come from somewhere else.
    
    # Let me check: in LAL, the tidal phase is computed by calling 
    # XLALSimNRTunedTidesFDTidalPhaseFrequencySeries with NRTidalv2NoAmpCorr_V
    # which calls SimNRTunedTidesFDTidalPhase_v2
    
    # But wait - in LAL's IMRPhenomD_NRTidal_Core, when NRTidal_version == NRTidalv2_V:
    #   ret = XLALSimNRTunedTidesFDTidalPhaseFrequencySeries(phi_tidal, amp_tidal, planck_taper, freqs, ..., NRTidalv2NoAmpCorr_V);
    #   XLALSimInspiralGetHOSpinTerms(&SS_3p5PN, &SSS_3p5PN, X_A, X_B, chi1, chi2, dquadmon1+1., dquadmon2+1.);
    #
    # And then:
    #   Corr = planck_taper->data[i] * cexp(-I*phi_tidal->data[i] - I*pn_fac*(SS_3p5PN + SSS_3p5PN)*pow(f,2./3.));
    #
    # So the total phase correction is: -(phi_tidal + pn_fac*(SS_3p5PN + SSS_3p5PN)*f^(2/3))
    #
    # But wait! LAL uses `pow(f, 2./3.)` not `pow(piM*f, 2./3.)`!
    # This is different from Ripple which uses x = (piM*f)^(2/3)!
    
    print(f"=== Spin phase frequency dependence ===")
    print(f"LAL: pn_fac * (SS_3p5PN + SSS_3p5PN) * f^(2/3)  [f in Hz]")
    print(f"Ripple: prefac * (SS_3p5 + SSS_3p5) * x  [x = (piM*f)^(2/3)]")
    print()
    print(f"LAL pn_fac = 3 * (piM)^(2/3) / (128 * eta)")
    print(f"Ripple prefac = 3 / (128 * eta)")
    print()
    print(f"LAL term = pn_fac * (SS+SSS) * f^(2/3)")
    print(f"         = 3*(piM)^(2/3)/(128*eta) * (SS+SSS) * f^(2/3)")
    print(f"         = 3/(128*eta) * (SS+SSS) * (piM*f)^(2/3)")
    print(f"         = prefac * (SS+SSS) * x")
    print()
    print(f"These are IDENTICAL!")
    print()
    
    # So the spin phase should match. Let me verify numerically:
    from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import get_spin_phase_correction
    
    chi1, chi2 = 0.02, -0.01
    theta = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    
    for f in test_freqs:
        M_s = M * MTSUN
        m1_s = m1 * MTSUN
        m2_s = m2 * MTSUN
        eta = m1_s * m2_s / (M_s**2.0)
        piM = PI * M_s
        
        # LAL style
        pn_fac = 3.0 * piM ** (2.0 / 3.0) / (128.0 * eta)
        
        # Compute SS_3p5PN and SSS_3p5PN (same as LAL)
        X_A = m1 / M
        X_B = m2 / M
        X_Asq = X_A**2
        X_Bsq = X_B**2
        chi1_sq = chi1**2
        chi2_sq = chi2**2
        
        # Quadparams
        from ripplegw.waveforms.IMRPhenom_tidal_utils import get_quadparam_octparam
        quadparam1, octparam1 = get_quadparam_octparam(jnp.array(lambda1))
        quadparam2, octparam2 = get_quadparam_octparam(jnp.array(lambda2))
        quadparam1 -= 1
        quadparam2 -= 1
        octparam1 -= 1
        octparam2 -= 1
        
        SS_3p5PN = (-400.0 * PI * float(quadparam1) * chi1_sq * X_Asq
                    - 400.0 * PI * float(quadparam2) * chi2_sq * X_Bsq)
        SSS_3p5PN = (
            10.0 * ((X_Asq + 308.0 / 3.0 * X_A) * chi1 + (X_Bsq - 89.0 / 3.0 * X_B) * chi2)
            * float(quadparam1) * X_Asq * chi1_sq
            + 10.0 * ((X_Bsq + 308.0 / 3.0 * X_B) * chi2 + (X_Asq - 89.0 / 3.0 * X_A) * chi1)
            * float(quadparam2) * X_Bsq * chi2_sq
            - 440.0 * float(octparam1) * X_A * X_Asq * chi1_sq * chi1
            - 440.0 * float(octparam2) * X_B * X_Bsq * chi2_sq * chi2
        )
        
        spin_lal = pn_fac * (SS_3p5PN + SSS_3p5PN) * f ** (2.0 / 3.0)
        
        # Ripple style
        x_val = jnp.array([(PI * M_s * f) ** (2.0 / 3.0)])
        spin_ripple = float(get_spin_phase_correction(x_val, theta)[0])
        
        print(f"f = {f} Hz:")
        print(f"  spin (LAL) = {spin_lal:.15e}")
        print(f"  spin (Ripple) = {spin_ripple:.15e}")
        print(f"  diff = {spin_lal - spin_ripple:.2e}")
        print()


if __name__ == "__main__":
    main()
