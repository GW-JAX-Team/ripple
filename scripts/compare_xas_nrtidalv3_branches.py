"""Compare XAS_NRTidalv3 overlap-loss distribution: current branch vs ripple-dev.

Generates n samples of LAL IMRPhenomXAS_NRTidalv3 waveforms, then computes the
per-sample noise-weighted overlap loss against ripple's implementation. The
ripple side is exercised twice via a file swap:
  (A) the current working-tree implementation
  (B) the implementation from the ripple-dev branch (loaded into a temp dir)

Saves both arrays as .npz and plots a side-by-side log10 histogram.

Run:
  uv run --group cross-validation python scripts/compare_xas_nrtidalv3_branches.py
"""

import os
import shutil
import subprocess
import sys
import importlib
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from tests.utils import (
    compute_overlap_loss,
    generate_random_params,
    get_freqs,
    get_lal_waveform,
    get_nyquist_mask,
)
from tests.cross_validation.test_lal_overlap import (
    DEFAULT_BOUNDS,
    convert_parameters_lal_to_ripple,
)


N_SAMPLES = 100
SEED = 42
T = 128.0
F_L, F_U, F_SAMPLING = 20.0, 4096.0, 8192.0
F_REF = 20.0
PSD_PATH = "tests/psds/ET_D_psd.txt"
CACHE_PATH = Path("baseline_results/xas_nrtidalv3_branch_compare_n100.npz")
OUT_FIG = Path("baseline_results/xas_nrtidalv3_branch_compare_n100.png")
PLOT_PATH_ROOT = "plot.png"
RIPPLE_DEV_SNAPSHOT_DIR = Path("baseline_results/_ripple_dev_snapshot")


SWAPPED_FILES = [
    "src/ripplegw/waveforms/IMRPhenomD_NRTidalv2.py",
    "src/ripplegw/waveforms/IMRPhenomXAS.py",
    "src/ripplegw/waveforms/IMRPhenomXAS_NRTidalv3.py",
    "src/ripplegw/waveforms/IMRPhenom_tidal_utils.py",
    "src/ripplegw/waveforms/NRTidalv3_utils.py",
]


def fetch_ripple_dev_files():
    """git-show each tracked file from ripple-dev to a snapshot directory."""
    RIPPLE_DEV_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for rel in SWAPPED_FILES:
        out_path = RIPPLE_DEV_SNAPSHOT_DIR / Path(rel).name
        with open(out_path, "wb") as f:
            subprocess.check_call(
                ["git", "show", f"ripple-dev:{rel}"], stdout=f
            )


def backup_current_files(backup_dir: Path):
    backup_dir.mkdir(parents=True, exist_ok=True)
    for rel in SWAPPED_FILES:
        shutil.copy2(rel, backup_dir / Path(rel).name)


def install_files_from(source_dir: Path):
    for rel in SWAPPED_FILES:
        shutil.copy2(source_dir / Path(rel).name, rel)


def reset_jax_and_modules():
    """Drop ripplegw modules so re-import picks up the swapped files."""
    for mod in list(sys.modules):
        if mod.startswith("ripplegw"):
            del sys.modules[mod]


def gen_lal_samples():
    """Generate (or load) 100 LAL XAS_NRTidalv3 waveforms."""
    if CACHE_PATH.exists():
        d = np.load(str(CACHE_PATH), allow_pickle=False)
        if int(d["hp_lal"].shape[0]) == N_SAMPLES:
            print(f"  [cache] loaded {N_SAMPLES} samples from {CACHE_PATH.name}")
            return (
                d["theta_batch"],
                [jnp.array(d["hp_lal"][i]) for i in range(N_SAMPLES)],
            )

    print(f"  [gen ] producing {N_SAMPLES} LAL waveforms ...", flush=True)
    fs = get_freqs(F_L, F_U, F_SAMPLING, T)
    df = float(fs[1] - fs[0])
    theta_batch = generate_random_params(
        N_SAMPLES, DEFAULT_BOUNDS, is_tidal=True, is_precessing=False, seed=SEED
    )
    hp_list_np = [None] * N_SAMPLES

    def _gen(i):
        hp_lal, _ = get_lal_waveform(
            theta_batch[i], "IMRPhenomXAS_NRTidalv3",
            F_L, F_U, df, F_REF, is_tidal=True, is_precessing=False,
        )
        return i, np.asarray(hp_lal)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, hp in pool.map(_gen, range(N_SAMPLES)):
            hp_list_np[i] = hp
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{N_SAMPLES}", flush=True)

    hp_arr = np.stack(hp_list_np, axis=0)
    np.savez(str(CACHE_PATH), theta_batch=theta_batch, hp_lal=hp_arr)
    print(f"  [save ] wrote {CACHE_PATH.name}")
    return theta_batch, [jnp.array(h) for h in hp_list_np]


def overlap_losses(theta_batch, hp_lal_list):
    """Compute per-sample overlap loss against ripplegw's current state."""
    reset_jax_and_modules()
    from tests.utils import get_jitted_waveform  # re-import after module reset

    fs = get_freqs(F_L, F_U, F_SAMPLING, T)
    psd_freqs, psd_vals = np.loadtxt(PSD_PATH, unpack=True)
    psd = jnp.interp(fs, jnp.array(psd_freqs), jnp.array(psd_vals))
    nyq_mask = get_nyquist_mask(fs, n_bins=2)
    waveform = get_jitted_waveform("IMRPhenomXAS_NRTidalv3", fs, F_REF)

    losses = np.zeros(N_SAMPLES)
    for i in range(N_SAMPLES):
        theta_rip = convert_parameters_lal_to_ripple(theta_batch[i], False, True)
        hp_rip, _ = waveform(theta_rip)
        hp_rip = hp_rip * nyq_mask
        hp_lal = hp_lal_list[i] * nyq_mask
        losses[i] = float(compute_overlap_loss(hp_lal, hp_rip, psd, fs))
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{N_SAMPLES}  last loss={losses[i]:.3e}", flush=True)
    return losses


def main():
    backup_dir = Path("baseline_results/_current_backup")
    fetch_ripple_dev_files()
    backup_current_files(backup_dir)

    # 1. LAL samples (single source of truth for both implementations)
    theta_batch, hp_lal_list = gen_lal_samples()

    # 2. Current branch overlap losses (we are on it now)
    print("\n[current branch] computing overlap losses ...", flush=True)
    losses_current = overlap_losses(theta_batch, hp_lal_list)
    print(f"  current branch: mean={losses_current.mean():.3e} "
          f"median={np.median(losses_current):.3e} max={losses_current.max():.3e}")

    # 3. Swap in ripple-dev files and recompute
    print("\n[ripple-dev]    swapping in ripple-dev sources ...", flush=True)
    install_files_from(RIPPLE_DEV_SNAPSHOT_DIR)
    try:
        losses_dev = overlap_losses(theta_batch, hp_lal_list)
        print(f"  ripple-dev:     mean={losses_dev.mean():.3e} "
              f"median={np.median(losses_dev):.3e} max={losses_dev.max():.3e}")
    finally:
        print("\n[restore] restoring current-branch files ...", flush=True)
        install_files_from(backup_dir)

    # 4. Persist arrays
    np.savez(
        "baseline_results/xas_nrtidalv3_branch_compare_losses.npz",
        losses_current=losses_current,
        losses_dev=losses_dev,
        theta_batch=theta_batch,
    )

    # 5. Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.linspace(-13, 0, 40)
    log_curr = np.log10(np.clip(losses_current, 1e-16, None))
    log_dev = np.log10(np.clip(losses_dev, 1e-16, None))
    ax.hist(
        log_dev, bins=bins, alpha=0.55, color="#d8513a", edgecolor="black",
        linewidth=0.6, label=f"ripple-dev (median {np.median(losses_dev):.2e})",
    )
    ax.hist(
        log_curr, bins=bins, alpha=0.55, color="#3a8acc", edgecolor="black",
        linewidth=0.6, label=f"this branch (median {np.median(losses_current):.2e})",
    )
    ax.axvline(np.log10(1e-6), ls="--", color="gray", lw=1)
    ax.text(np.log10(1e-6) + 0.05, ax.get_ylim()[1] * 0.95, "test threshold 1e-6",
            color="gray", fontsize=9, va="top")
    ax.set_xlabel(r"$\log_{10}(\mathrm{overlap\ loss})$ vs LAL")
    ax.set_ylabel("count")
    ax.set_title(
        f"IMRPhenomXAS_NRTidalv3 vs LAL — N={N_SAMPLES}, ET-D PSD, T=128s"
    )
    ax.legend(loc="upper left", framealpha=0.92)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(OUT_FIG), dpi=150)
    fig.savefig(PLOT_PATH_ROOT, dpi=150)
    print(f"\n[fig] wrote {OUT_FIG} and {PLOT_PATH_ROOT}")


if __name__ == "__main__":
    main()
