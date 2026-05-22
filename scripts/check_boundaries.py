#!/usr/bin/env python
"""Check phase region boundaries for typical BNS parameters."""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ripplegw.waveforms.IMRPhenomX_utils import PhenomX_phase_coeff_table, get_cutoff_fMs
from ripplegw.waveforms.NRTidalv3_utils import _get_merger_frequency
from ripplegw.constants import MTSUN, PI

# Test case 1: BNS with lambda=1000
m1, m2 = 1.4, 1.4
chi1, chi2 = 0.0, 0.0
lambda1, lambda2 = 1000.0, 1000.0
M_s = (m1 + m2) * MTSUN
theta_intrinsic = jnp.array([m1, m2, chi1, chi2, lambda1, lambda2])

fMs_RD, fMs_damp, fMs_2, fMs_1 = get_cutoff_fMs(m1, m2, chi1, chi2)
print(f"BNS (m1=m2=1.4, lambda=1000):")
print(f"  fMs_1 (inspiral top) = {float(fMs_1):.6f}, f_1 = {float(fMs_1)/M_s:.2f} Hz")
print(f"  fMs_2 (intermediate top) = {float(fMs_2):.6f}, f_2 = {float(fMs_2)/M_s:.2f} Hz")
print(f"  fMs_RD (ringdown) = {float(fMs_RD):.6f}, f_RD = {float(fMs_RD)/M_s:.2f} Hz")
print(f"  fMs_damp = {float(fMs_damp):.6f}, f_damp = {float(fMs_damp)/M_s:.2f} Hz")
print(f"  f_merger = {float(_get_merger_frequency(theta_intrinsic)):.2f} Hz")
f_merger = float(_get_merger_frequency(theta_intrinsic))
print(f"  f_merger * M_s = {f_merger * M_s:.6f}")
print(f"  f_merger region: {'inspiral' if f_merger*M_s < float(fMs_1) else 'intermediate' if f_merger*M_s < float(fMs_2) else 'ringdown'}")
print()

# Test case 2: BNS with lambda=0
theta_intrinsic0 = jnp.array([m1, m2, chi1, chi2, 0.0, 0.0])
f_merger0 = float(_get_merger_frequency(theta_intrinsic0))
print(f"BNS (m1=m2=1.4, lambda=0):")
print(f"  f_merger = {f_merger0:.2f} Hz, Mf_merger = {f_merger0 * M_s:.6f}")
print(f"  f_merger region: {'inspiral' if f_merger0*M_s < float(fMs_1) else 'intermediate' if f_merger0*M_s < float(fMs_2) else 'ringdown'}")
print()

# Test case 3: BBH-like BNS range
m1b, m2b = 2.5, 1.0
chi1b, chi2b = 0.0, 0.0
M_s_b = (m1b + m2b) * MTSUN
fMs_RD_b, fMs_damp_b, fMs_2_b, fMs_1_b = get_cutoff_fMs(m1b, m2b, chi1b, chi2b)
theta_intrinsic_b = jnp.array([m1b, m2b, chi1b, chi2b, 500.0, 200.0])
f_merger_b = float(_get_merger_frequency(theta_intrinsic_b))
print(f"BNS (m1=2.5, m2=1.0, lambda1=500, lambda2=200):")
print(f"  fMs_1 = {float(fMs_1_b):.6f}, f_1 = {float(fMs_1_b)/M_s_b:.2f} Hz")
print(f"  fMs_2 = {float(fMs_2_b):.6f}, f_2 = {float(fMs_2_b)/M_s_b:.2f} Hz")
print(f"  f_merger = {f_merger_b:.2f} Hz, Mf_merger = {f_merger_b * M_s_b:.6f}")
print(f"  f_merger region: {'inspiral' if f_merger_b*M_s_b < float(fMs_1_b) else 'intermediate' if f_merger_b*M_s_b < float(fMs_2_b) else 'ringdown'}")
print()

# Key: what fraction of random BNS parameters (from DEFAULT_BOUNDS) have f_merger in the intermediate region?
# Let's scan a range of parameters
import numpy as np
np.random.seed(42)
n = 1000
m1_arr = np.random.uniform(0.5, 3.0, n)
m2_arr = np.random.uniform(0.5, 3.0, n)
chi1_arr = np.random.uniform(-0.05, 0.05, n)
chi2_arr = np.random.uniform(-0.05, 0.05, n)
lambda1_arr = np.random.uniform(0.0, 5000.0, n)
lambda2_arr = np.random.uniform(0.0, 5000.0, n)

regions = {'inspiral': 0, 'intermediate': 0, 'ringdown': 0}
fMs_finals = []

for i in range(n):
    m1_i, m2_i = m1_arr[i], m2_arr[i]
    chi1_i, chi2_i = chi1_arr[i], chi2_arr[i]
    l1_i, l2_i = lambda1_arr[i], lambda2_arr[i]
    M_s_i = (m1_i + m2_i) * MTSUN

    theta_i = jnp.array([m1_i, m2_i, chi1_i, chi2_i, l1_i, l2_i])
    f_merger_i = float(_get_merger_frequency(theta_i))
    Mf_merger_i = f_merger_i * M_s_i
    fMs_RD_i, fMs_damp_i, fMs_2_i, fMs_1_i = get_cutoff_fMs(m1_i, m2_i, chi1_i, chi2_i)
    fMs_1_i, fMs_2_i = float(fMs_1_i), float(fMs_2_i)

    fMs_finals.append(Mf_merger_i)

    if Mf_merger_i < fMs_1_i:
        regions['inspiral'] += 1
    elif Mf_merger_i < fMs_2_i:
        regions['intermediate'] += 1
    else:
        regions['ringdown'] += 1

fMs_finals = np.array(fMs_finals)
print(f"Distribution of f_merger region (n={n} samples):")
print(f"  Inspiral: {regions['inspiral']} ({100*regions['inspiral']/n:.1f}%)")
print(f"  Intermediate: {regions['intermediate']} ({100*regions['intermediate']/n:.1f}%)")
print(f"  Ringdown: {regions['ringdown']} ({100*regions['ringdown']/n:.1f}%)")
print(f"  Mf_final distribution: min={fMs_finals.min():.4f}, mean={fMs_finals.mean():.4f}, max={fMs_finals.max():.4f}")
