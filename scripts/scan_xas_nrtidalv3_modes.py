"""Test the impact of different XAS_NRTidalv3 mode settings on overlap loss vs LAL.

Tests combinations of:
  RIPPLE_XAS_FFINAL_MODE: "last" (LAL-faithful) vs "plus_df" (current default)
  RIPPLE_XAS_DPHIXAS_MODE: "analytic" (LAL-faithful) vs "secant" (current default)
  RIPPLE_XAS_DPHIT_MODE:   "analytic" (LAL-faithful) vs "secant"

Result printed as a small table.
"""

import os
import sys
import itertools

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# IMPORTANT: set env vars BEFORE importing ripplegw (defaults captured at import)
def run_one(ffinal_mode, dphixas_mode, dphit_mode):
    os.environ["RIPPLE_XAS_FFINAL_MODE"] = ffinal_mode
    os.environ["RIPPLE_XAS_DPHIXAS_MODE"] = dphixas_mode
    os.environ["RIPPLE_XAS_DPHIT_MODE"] = dphit_mode

    # Force fresh import of the XAS_NRTidalv3 module so env vars take effect
    for mod in list(sys.modules):
        if "IMRPhenomXAS_NRTidalv3" in mod:
            del sys.modules[mod]

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from pathlib import Path

    from tests.utils import (
        compute_overlap_loss,
        get_freqs,
        get_jitted_waveform,
        get_nyquist_mask,
    )
    from tests.cross_validation.test_lal_overlap import convert_parameters_lal_to_ripple

    # Match the cache parameters: T=128, f_l=20, f_u=4096, f_sampling=8192
    T = 128.0
    f_l, f_u, f_sampling = 20.0, 4096.0, 8192.0
    f_ref = 20.0

    fs = get_freqs(f_l, f_u, f_sampling, T)
    df = float(fs[1] - fs[0])

    # Load cache
    cache_path = Path("tests/cross_validation/lal_cache/IMRPhenomXAS_NRTidalv3_T128.npz")
    data = np.load(str(cache_path), allow_pickle=False)
    theta_batch = data["theta_batch"]
    hp_lal_all = data["hp_lal"]
    hc_lal_all = data["hc_lal"]
    valid_mask = data["valid_mask"].astype(bool)

    n_samples = 10
    valid_idx = np.where(valid_mask[:n_samples])[0]

    # PSD
    psd_path = Path("tests/psds/ET_D_psd.txt")
    psd_freqs, psd_vals = np.loadtxt(psd_path, unpack=True)
    psd = jnp.interp(fs, jnp.array(psd_freqs), jnp.array(psd_vals))

    nyquist_mask = get_nyquist_mask(fs, n_bins=2)

    waveform = get_jitted_waveform("IMRPhenomXAS_NRTidalv3", fs, f_ref)

    losses = []
    for i in valid_idx:
        theta_ripple = convert_parameters_lal_to_ripple(theta_batch[i], False, True)
        hp_rip, _ = waveform(theta_ripple)
        hp_rip = hp_rip * nyquist_mask
        hp_lal = jnp.array(hp_lal_all[i]) * nyquist_mask
        loss = float(compute_overlap_loss(hp_lal, hp_rip, psd, fs))
        losses.append(loss)

    losses = np.array(losses)
    return float(losses.mean()), float(np.median(losses)), float(losses.max())


def main():
    combos = list(itertools.product(
        ["plus_df", "last"],     # FFINAL
        ["secant", "analytic"],  # DPHIXAS
        ["analytic", "secant"],  # DPHIT
    ))

    print(f"{'FFINAL':<10} {'DPHIXAS':<10} {'DPHIT':<10} {'mean':>12} {'median':>12} {'max':>12}")
    print("-" * 80)
    for ffinal, dxas, dt in combos:
        try:
            mean, median, mx = run_one(ffinal, dxas, dt)
            print(f"{ffinal:<10} {dxas:<10} {dt:<10} {mean:12.3e} {median:12.3e} {mx:12.3e}")
        except Exception as e:
            print(f"{ffinal:<10} {dxas:<10} {dt:<10}  ERROR: {e}")


if __name__ == "__main__":
    main()
