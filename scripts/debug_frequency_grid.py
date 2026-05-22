#!/usr/bin/env python
"""Check if LAL and Ripple frequency grids align."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils import get_freqs

jax.config.update("jax_enable_x64", True)


def main():
    f_l, f_u, f_sampling, T, f_ref = 20.0, 4096.0, 8192.0, 128.0, 20.0
    
    # Ripple's frequency grid
    fs = get_freqs(f_l, f_u, f_sampling, T)
    fs_np = np.array(fs)
    df_ripple = float(fs_np[1] - fs_np[0])
    
    print(f"Ripple frequency grid:")
    print(f"  df = {df_ripple:.15e} Hz")
    print(f"  f[0] = {float(fs_np[0]):.15e} Hz")
    print(f"  f[1] = {float(fs_np[1]):.15e} Hz")
    print(f"  f[-1] = {float(fs_np[-1]):.15e} Hz")
    print(f"  n_freqs = {len(fs_np)}")
    print()
    
    # LAL's frequency grid (uniform, starting from 0)
    delta_t = 1.0 / f_sampling
    tlen = int(round(T / delta_t))
    print(f"LAL parameters:")
    print(f"  delta_t = {delta_t:.15e} s")
    print(f"  tlen = {tlen}")
    print(f"  df_LAL = {1.0 / (tlen * delta_t):.15e} Hz")
    print()
    
    # LAL's frequency array
    freqs_lal = np.fft.rfftfreq(tlen, delta_t)
    mask_lal = (freqs_lal > f_l) & (freqs_lal < f_u)
    fs_lal = freqs_lal[mask_lal]
    
    print(f"LAL frequency grid (masked to f_l, f_u):")
    print(f"  df = {fs_lal[1] - fs_lal[0]:.15e} Hz")
    print(f"  f[0] = {fs_lal[0]:.15e} Hz")
    print(f"  f[1] = {fs_lal[1]:.15e} Hz")
    print(f"  f[-1] = {fs_lal[-1]:.15e} Hz")
    print(f"  n_freqs = {len(fs_lal)}")
    print()
    
    # Check if grids match
    print(f"=== Grid comparison ===")
    print(f"  n_ripple = {len(fs_np)}, n_lal = {len(fs_lal)}")
    print(f"  df_ripple = {df_ripple:.15e}, df_lal = {fs_lal[1] - fs_lal[0]:.15e}")
    print(f"  df diff = {df_ripple - (fs_lal[1] - fs_lal[0]):.2e}")
    print()
    
    if len(fs_np) == len(fs_lal):
        f_diff = fs_np - fs_lal
        print(f"  Frequency differences:")
        print(f"    Max abs diff = {np.max(np.abs(f_diff)):.2e}")
        print(f"    f[0] diff = {f_diff[0]:.2e}")
        print(f"    f[-1] diff = {f_diff[-1]:.2e}")
    else:
        print(f"  Grids have different lengths!")
        print(f"  This means LAL and Ripple use different frequency bins!")
    
    # Check if the issue is in how get_lal_waveform extracts frequencies
    # LAL's output has frequencies i*df for i = 0, 1, 2, ...
    # The test extracts data where (freqs_lal > f_l) & (freqs_lal < f_u)
    # But Ripple's frequencies are from rfftfreq with the same tlen and delta_t
    # So they should match!
    
    # Let me verify by computing the expected LAL frequencies
    delta_f = 1.0 / (tlen * delta_t)
    print(f"Expected LAL df: {delta_f:.15e}")
    print(f"  1/T = {1.0/T:.15e}")
    print(f"  df_ripple = {df_ripple:.15e}")
    print(f"  Match: {abs(delta_f - df_ripple) < 1e-15}")
    
    # Actually, df = 1/(tlen * delta_t) = 1/T (approximately)
    # But tlen = round(T / delta_t), so df = 1/(round(T/delta_t) * delta_t) != 1/T exactly
    
    tlen_exact = T / delta_t
    df_exact = 1.0 / (tlen_exact * delta_t)
    df_actual = 1.0 / (tlen * delta_t)
    
    print(f"\n  tlen_exact = T/delta_t = {tlen_exact:.15e}")
    print(f"  tlen_rounded = {tlen}")
    print(f"  df_exact = 1/T = {df_exact:.15e}")
    print(f"  df_actual = 1/(tlen*delta_t) = {df_actual:.15e}")
    print(f"  df diff = {df_actual - df_exact:.2e}")
    
    # The df difference could cause phase errors!
    # Phase at frequency f is phi(f). If the frequency is off by df_err,
    # the phase error is dphi/df * df_err.
    # For the tidal phase, dphi/df ~ psi_T / f.
    # At 500 Hz, psi_T ~ -1.24, so dphi/df ~ -1.24/500 ~ -0.0025 rad/Hz.
    # If df_err ~ 1e-10 Hz, then phase error ~ 2.5e-13 rad. Too small.
    
    # So the frequency grid alignment is not the issue.


if __name__ == "__main__":
    main()
