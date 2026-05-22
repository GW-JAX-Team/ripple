"""Compare ripple's FULL BBH XAS waveform phase (with alignment) vs LAL phi(f)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import lal
import lalsimulation as lalsim
from ripplegw.constants import PI, MTSUN
from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphc
from ripplegw.conversions import ms_to_Mc_eta


def main():
    m1, m2 = 2.9248, 2.8768
    s1z, s2z = -0.0329, -0.0361
    dL = 157.5
    M_s = (m1 + m2) * MTSUN

    T = 128.0
    df = 1.0 / T
    f_l, f_u, f_ref = 20.0, 4096.0, 20.0

    # LAL phi(f)
    p_dict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPhenomXOnlyReturnPhase(p_dict, 1)
    hp, _ = lalsim.SimInspiralChooseFDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0.0, 0.0, s1z, 0.0, 0.0, s2z,
        dL*1e6*lal.PC_SI, 0.0, 0.0, 0.0, 0.0, 0.0,
        df, f_l, f_u, f_ref, p_dict, lalsim.IMRPhenomXAS,
    )
    lal_freqs = np.arange(hp.data.length) * hp.deltaF
    band = (lal_freqs >= f_l) & (lal_freqs < f_u)
    lal_phi = np.asarray(hp.data.data)[band].real
    fb = lal_freqs[band]

    # Ripple BBH XAS full waveform → extract angle
    # We need a clean iota=0 plus polarization (h+ = h0/2 * (1+cos2(iota)))
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    rp = jnp.array([Mc, eta, s1z, s2z, dL, 0.0, 0.0, 0.0])
    rip_hp, _ = gen_IMRPhenomXAS_hphc(jnp.array(fb), rp, f_ref)
    rip_phi = np.unwrap(np.angle(np.array(rip_hp)))

    print(f"LAL phi @ fRef:    {lal_phi[0]:.6f}")
    print(f"Ripple phi @ fRef: {rip_phi[0]:.6f}")
    print(f"LAL phi @ 100Hz:   {lal_phi[np.argmin(np.abs(fb-100))]:.6f}")
    print(f"Ripple phi @ 100Hz:{rip_phi[np.argmin(np.abs(fb-100))]:.6f}")
    print(f"LAL phi @ 1000Hz:  {lal_phi[np.argmin(np.abs(fb-1000))]:.6f}")
    print(f"Ripple phi @ 1000Hz:{rip_phi[np.argmin(np.abs(fb-1000))]:.6f}")

    # Wrap LAL phi modulo 2pi for comparison
    lal_wrapped = ((lal_phi + PI) % (2*PI)) - PI
    rip_wrapped = ((rip_phi + PI) % (2*PI)) - PI
    # Difference (in wrapped space)
    diff_wrap = lal_wrapped - rip_wrapped
    diff_wrap = ((diff_wrap + PI) % (2*PI)) - PI
    print(f"\nWrapped diff (mod 2pi): max|d|={np.abs(diff_wrap).max():.3e}"
          f"  RMS={np.sqrt(np.mean(diff_wrap**2)):.3e}")
    bands = [(20, 50), (50, 100), (100, 200), (200, 500),
             (500, 1000), (1000, 2000), (2000, 4000)]
    for lo, hi in bands:
        m = (fb >= lo) & (fb < hi)
        if m.any():
            print(f"  {lo:>4}-{hi:<4} Hz: max|d|={np.abs(diff_wrap[m]).max():.3e}")


if __name__ == "__main__":
    main()
