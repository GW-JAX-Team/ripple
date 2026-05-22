"""Compare ripple's PURE BBH IMRPhenomXAS phase vs LAL's PhenomXOnlyReturnPhase.

This bypasses the tidal correction entirely. If BBH-only phases match well,
then the residual in the NRTidalv3 variant must come from the tidal alignment
(linb, phiTfRef) rather than the underlying BBH phase.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import lal
import lalsimulation as lalsim

from ripplegw.constants import MTSUN, PI
from ripplegw.waveforms import IMRPhenomX_utils
from ripplegw.waveforms.IMRPhenomXAS import Phase


def main():
    m1, m2 = 2.9248, 2.8768
    s1z, s2z = -0.0329, -0.0361
    dL = 157.5

    M_s = (m1 + m2) * MTSUN
    theta_bbh = jnp.array([m1, m2, s1z, s2z])

    T = 128.0
    df = 1.0 / T
    f_l, f_u, f_ref = 20.0, 4096.0, 20.0

    # ---- LAL BBH XAS ----
    p_dict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXOnlyReturnPhase(p_dict, 1)
    hp, _ = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0.0, 0.0, s1z, 0.0, 0.0, s2z,
        dL * 1e6 * lal.PC_SI, 0.0, 0.0, 0.0, 0.0, 0.0,
        df, f_l, f_u, f_ref, p_dict, lalsim.IMRPhenomXAS,
    )
    lal_freqs = np.arange(hp.data.length) * hp.deltaF
    band = (lal_freqs >= f_l) & (lal_freqs < f_u)
    lal_phi = np.asarray(hp.data.data)[band].real
    fb = lal_freqs[band]
    print(f"LAL phi(f) sample range: [{lal_phi.min():.3f}, {lal_phi.max():.3f}]")
    print(f"LAL phi at f=20Hz:  {lal_phi[0]:.6f}")
    print(f"LAL phi at f=100Hz: {lal_phi[np.argmin(np.abs(fb-100))]:.6f}")

    # ---- Ripple BBH XAS Phase ----
    coeffs = IMRPhenomX_utils.PhenomX_phase_coeff_table
    ripple_phase = np.array(Phase(jnp.array(fb), theta_bbh, coeffs))
    print(f"\nRipple Phase(f) sample range: [{ripple_phase.min():.3f}, {ripple_phase.max():.3f}]")
    print(f"Ripple at 20Hz:  {ripple_phase[0]:.6f}")
    print(f"Ripple at 100Hz: {ripple_phase[np.argmin(np.abs(fb-100))]:.6f}")

    # Subtract value at fRef first (proxy for "remove phifRef constant")
    iref = np.argmin(np.abs(fb - f_ref))
    ripple_norm = ripple_phase - ripple_phase[iref]
    lal_norm = lal_phi - lal_phi[iref]
    diff = ripple_norm - lal_norm
    A = np.vstack([fb - f_ref, np.ones_like(fb)]).T
    coef, *_ = np.linalg.lstsq(A, diff, rcond=None)
    resid = diff - A @ coef
    print(f"\n[After subtracting value at fRef]")
    print(f"  ripple_norm at 100Hz: {ripple_norm[np.argmin(np.abs(fb-100))]:.6f}")
    print(f"  lal_norm at 100Hz:    {lal_norm[np.argmin(np.abs(fb-100))]:.6f}")
    print(f"\n(ripple Phase) - (LAL phi):")
    print(f"  best-fit slope: {coef[0]:.6e} rad/Hz  (=> tshift {coef[0]/(2*PI):.3e} s)")
    print(f"  best-fit const: {coef[1]:.6e} rad")
    print(f"  residual: max|d|={np.abs(resid).max():.3e}"
          f"  RMS={np.sqrt(np.mean(resid**2)):.3e}")
    bands = [(20, 50), (50, 100), (100, 200), (200, 500),
             (500, 1000), (1000, 2000), (2000, 4000)]
    for lo, hi in bands:
        m = (fb >= lo) & (fb < hi)
        if m.any():
            print(f"    {lo:>4}-{hi:<4} Hz: max|resid|={np.abs(resid[m]).max():.3e}"
                  f"  RMS={np.sqrt(np.mean(resid[m]**2)):.3e}")


if __name__ == "__main__":
    main()
