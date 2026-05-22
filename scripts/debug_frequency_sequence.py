#!/usr/bin/env python
"""Run a cleaner benchmark using LAL's FrequencySequence API to avoid truncation artifacts."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ripplegw.constants import MTSUN, PI
from ripplegw.conversions import ms_to_Mc_eta, lambdas_to_lambda_tildes
from tests.utils import (
    get_freqs,
    get_jitted_waveform,
    compute_overlap_loss,
    generate_random_params,
)
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import _get_merger_frequency

jax.config.update("jax_enable_x64", True)

# Check if LALSuite is available
try:
    import lal
    import lalsimulation as lalsim
    LAL_AVAILABLE = True
except ImportError:
    LAL_AVAILABLE = False


def get_lal_waveform_frequency_sequence(theta, f_l, f_u, fs_np, f_ref, is_tidal):
    """Generate LAL waveform using FrequencySequence API (no truncation)."""
    if not LAL_AVAILABLE:
        raise ImportError("LALSuite required")
    
    approximant = lalsim.SimInspiralGetApproximantFromString("IMRPhenomD_NRTidalv2")
    
    m1_kg = theta[0] * lal.MSUN_SI
    m2_kg = theta[1] * lal.MSUN_SI
    s1z = theta[2]
    s2z = theta[3]
    l1 = theta[4]
    l2 = theta[5]
    distance = theta[6] * 1e6 * lal.PC_SI
    phi_ref = theta[8]
    inclination = theta[9]
    
    laldict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(laldict, l1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(laldict, l2)
    quad1 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l1)
    quad2 = lalsim.SimUniversalRelationQuadMonVSlambda2Tidal(l2)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon1(laldict, quad1 - 1)
    lalsim.SimInspiralWaveformParamsInsertdQuadMon2(laldict, quad2 - 1)
    
    # Create frequency sequence
    freqs_seq = lal.CreateREAL8Sequence(len(fs_np))
    for i, f in enumerate(fs_np):
        freqs_seq.data[i] = float(f)
    
    hp = lalsim.SimIMRPhenomDNRTidalFrequencySequence(
        freqs_seq, phi_ref, f_ref, distance,
        m1_kg, m2_kg, s1z, s2z, l1, l2, laldict,
        lalsim.NRTidalv2_V
    )
    
    return np.array(hp.data.data), np.array(hp.data.data)  # hplus = hcross for aligned spin (up to phase)


def main():
    # Use the same test case
    m1, m2 = 1.4, 1.3
    chi1, chi2 = 0.02, -0.01
    lambda1, lambda2 = 400.0, 300.0
    dist_mpc = 100.0
    tc = 0.0
    phic = 0.0
    inclination = np.pi / 4
    
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(
        jnp.array([lambda1, lambda2, m1, m2])
    )
    theta_ripple = jnp.array([
        Mc, eta, chi1, chi2, lambda_tilde, delta_lambda_tilde,
        dist_mpc, tc, phic, inclination
    ])
    theta_lal = np.array([m1, m2, chi1, chi2, lambda1, lambda2, dist_mpc, tc, phic, inclination])
    
    # Use frequency grid up to f_merger * 1.3 (NRTIDAL_FMAX) to avoid truncation issues
    f_l, f_u, f_sampling, T, f_ref = 20.0, 4096.0, 8192.0, 128.0, 20.0
    fs = get_freqs(f_l, f_u, f_sampling, T)
    fs_np = np.array(fs)
    
    # Compute merger frequency
    theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])
    f_merger = float(_get_merger_frequency(theta_intrinsic))
    NRTIDAL_FMAX = 1.3 * f_merger
    
    print(f"f_merger = {f_merger:.1f} Hz")
    print(f"NRTIDAL_FMAX = {NRTIDAL_FMAX:.1f} Hz")
    print()
    
    # Generate LAL waveform using FrequencySequence
    hp_lal, hc_lal = get_lal_waveform_frequency_sequence(
        theta_lal, f_l, f_u, fs_np, f_ref, is_tidal=True
    )
    
    # Generate Ripple waveform
    waveform = get_jitted_waveform("IMRPhenomD_NRTidalv2", fs, f_ref)
    hp_ripple, hc_ripple = waveform(theta_ripple)
    hp_ripple_np = np.array(hp_ripple)
    
    # Compare
    print(f"=== Waveform comparison (FrequencySequence) ===")
    test_freqs = np.array([50.0, 500.0, 1000.0, 2000.0, 2500.0])
    for tf in test_freqs:
        idx = np.argmin(np.abs(fs_np - tf))
        amp_lal = np.abs(hp_lal[idx])
        amp_rip = np.abs(hp_ripple_np[idx])
        phase_lal = np.angle(hp_lal[idx])
        phase_rip = np.angle(hp_ripple_np[idx])
        print(f"  f={tf:.0f} Hz:")
        print(f"    |hp| LAL = {amp_lal:.15e}, Ripple = {amp_rip:.15e}, ratio = {amp_rip/amp_lal:.15f}")
        print(f"    phase LAL = {phase_lal:.15e}, Ripple = {phase_rip:.15e}, diff = {phase_rip - phase_lal:.15e}")
    print()
    
    # Compute overlap loss
    psd_path = Path(__file__).parent.parent / "tests" / "psds" / "ET_D_psd.txt"
    psd_freqs_np, psd_np = np.loadtxt(psd_path, unpack=True)
    psd_interp = jnp.interp(fs, jnp.array(psd_freqs_np), jnp.array(psd_np))
    
    from tests.utils import get_nyquist_mask
    nyquist_mask = get_nyquist_mask(fs)
    
    overlap_loss = compute_overlap_loss(
        hp_ripple * nyquist_mask, jnp.array(hp_lal) * nyquist_mask,
        psd_interp, fs
    )
    print(f"Overlap loss: {float(overlap_loss):.15e}")
    print(f"log10(overlap_loss): {np.log10(float(overlap_loss)):.4f}")


if __name__ == "__main__":
    main()
