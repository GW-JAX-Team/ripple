"""Baseline mismatch run for IMRPhenomXHM / IMRPhenomXPHM.

Reproducible (seed=42) parameter sets covering the full BBH range are
swept; for each, ripple and LAL produce hp/hc on the same frequency
grid and the noise-weighted match is computed.  Outputs:

  baseline_results/<run_tag>/<waveform>_samples.npz
      Arrays per sample:
        - theta : (N, n_params)  parameter values used (LAL convention)
        - mismatch_hp / _hc : (N,)  1 - match per polarisation
        - mismatch  : (N,)  max(mismatch_hp, mismatch_hc)
        - msa_fallback_mask : (N,) bool  XPHM samples LAL refused (NaN mismatch)

  baseline_results/<run_tag>/<waveform>_summary.json
      Summary statistics (min/max/mean/median/quantiles) computed over
      finite mismatch values.

Re-run with the same `--n-samples` / `--T` after each fix and diff the
summary JSONs to track progress.

Usage:
    uv run --group cross-validation python scripts/baseline_xhm_xphm_mismatch.py \
        --n-samples 50 --tag pre_fix
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.utils import (  # noqa: E402
    LAL_AVAILABLE,
    compute_match,
    generate_random_params,
    get_freqs,
    get_jitted_waveform,
    get_lal_waveform,
    get_nyquist_mask,
)

# Reuse the helpers that test_lal_mismatch uses to keep the parameter
# convention identical across both code paths.
from tests.cross_validation.test_lal_mismatch import (  # noqa: E402
    BBH_BOUNDS,
    convert_parameters_lal_to_ripple,
)


# Match the test_lal_mismatch defaults.
WAVEFORMS = ["IMRPhenomXHM", "IMRPhenomXPHM"]
DEFAULT_N_SAMPLES = 30
DEFAULT_T = 32.0
F_L = 20.0
F_U = 2048.0
F_REF = 20.0
F_SAMPLING = 2 * F_U
SEED = 42


def _summary_stats(values: np.ndarray, threshold: float) -> dict:
    finite = np.isfinite(values)
    n_total = int(values.size)
    n_finite = int(finite.sum())
    n_failed = int(np.sum(values[finite] > threshold))
    if n_finite == 0:
        return dict(
            n_total=n_total,
            n_finite=0,
            n_excluded=n_total,
            n_failed=0,
            min=None,
            max=None,
            mean=None,
            median=None,
            q25=None,
            q75=None,
            q95=None,
            threshold=threshold,
            passed=False,
        )
    v = values[finite]
    return dict(
        n_total=n_total,
        n_finite=n_finite,
        n_excluded=n_total - n_finite,
        n_failed=n_failed,
        min=float(np.min(v)),
        max=float(np.max(v)),
        mean=float(np.mean(v)),
        median=float(np.median(v)),
        q25=float(np.quantile(v, 0.25)),
        q75=float(np.quantile(v, 0.75)),
        q95=float(np.quantile(v, 0.95)),
        threshold=threshold,
        passed=bool(n_failed == 0),
    )


def _run_lal_batch(theta_batch: np.ndarray, waveform_name: str, df: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute LAL waveforms for every sample in parallel.

    Returns (hp_store, hc_store, msa_fallback_mask).  When LAL refuses a
    sample (XPHM PrecVersion=222 MSA failure), the corresponding row is
    NaN and the fallback mask flags it.
    """
    n = theta_batch.shape[0]
    n_freqs_grid = int(round((F_U - F_L) * 1.0 / df))  # rough placeholder; resized below
    is_precessing = waveform_name == "IMRPhenomXPHM"

    def _one(i_theta):
        i, theta = i_theta
        try:
            hp, hc = get_lal_waveform(
                theta,
                waveform_name,
                F_L,
                F_U,
                df,
                F_REF,
                False,  # is_tidal
                is_precessing,
            )
            return i, hp, hc, False, None
        except Exception as exc:
            is_msa = is_precessing
            return i, None, None, is_msa, str(exc)

    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu = os.cpu_count() or 1
    n_workers = min(n, n_cpu)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_one, enumerate(theta_batch)))

    # Determine the actual frequency-grid length from the first successful sample.
    n_freqs = None
    for _, hp, _, _, err in results:
        if err is None:
            n_freqs = len(hp)
            break
    if n_freqs is None:
        raise RuntimeError(f"No successful LAL waveforms for {waveform_name}")

    hp_store = np.full((n, n_freqs), np.nan, dtype=np.complex128)
    hc_store = np.full((n, n_freqs), np.nan, dtype=np.complex128)
    msa_mask = np.zeros(n, dtype=bool)
    for i, hp, hc, is_msa, err in results:
        if err is None:
            hp_store[i] = hp
            hc_store[i] = hc
        elif is_msa:
            msa_mask[i] = True
        else:
            print(f"  [WARN] sample {i} raised: {err}")
    return hp_store, hc_store, msa_mask


def run_waveform(waveform_name: str, theta_batch: np.ndarray, n_samples: int,
                 T: float, threshold: float, out_dir: Path) -> dict:
    is_precessing = waveform_name == "IMRPhenomXPHM"
    fs = get_freqs(F_L, F_U, F_SAMPLING, T)
    df = float(fs[1] - fs[0])
    print(f"\n=== {waveform_name} : {n_samples} samples, T={T}s, n_freqs={len(fs)} ===")

    # ----- LAL -----
    t0 = time.time()
    hp_lal_arr, hc_lal_arr, msa_mask = _run_lal_batch(theta_batch, waveform_name, df)
    print(f"  LAL done in {time.time() - t0:.1f}s "
          f"(MSA fallback: {int(msa_mask.sum())} / {n_samples})")

    # ----- Ripple (jitted, vectorised manually to keep memory bounded) -----
    t0 = time.time()
    waveform_fn = get_jitted_waveform(waveform_name, fs, F_REF)
    nyquist_mask = get_nyquist_mask(fs)

    psd_path = REPO_ROOT / "tests" / "psds" / "ET_D_psd.txt"
    psd_freqs, psd = np.loadtxt(psd_path, unpack=True)
    psd_interp = jnp.interp(fs, jnp.array(psd_freqs), jnp.array(psd))

    mismatch_hp = np.full(n_samples, np.nan)
    mismatch_hc = np.full(n_samples, np.nan)

    for i in range(n_samples):
        if msa_mask[i] or not np.all(np.isfinite(hp_lal_arr[i])):
            continue
        theta_lal = theta_batch[i]
        theta_rip = convert_parameters_lal_to_ripple(theta_lal, is_precessing, False)
        hp_rip, hc_rip = waveform_fn(theta_rip)
        hp_rip = hp_rip * nyquist_mask
        hc_rip = hc_rip * nyquist_mask
        hp_l = jnp.array(hp_lal_arr[i]) * nyquist_mask
        hc_l = jnp.array(hc_lal_arr[i]) * nyquist_mask
        m_hp = float(compute_match(hp_rip, hp_l, psd_interp, fs))
        m_hc = float(compute_match(hc_rip, hc_l, psd_interp, fs))
        mismatch_hp[i] = 1.0 - m_hp
        mismatch_hc[i] = 1.0 - m_hc
    print(f"  Ripple match done in {time.time() - t0:.1f}s")

    mismatch = np.maximum(mismatch_hp, mismatch_hc)

    # ----- Save -----
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{waveform_name}_samples.npz"
    np.savez(
        str(npz_path),
        theta=theta_batch,
        mismatch_hp=mismatch_hp,
        mismatch_hc=mismatch_hc,
        mismatch=mismatch,
        msa_fallback_mask=msa_mask,
        seed=np.array(SEED),
        T=np.array(T),
        f_l=np.array(F_L),
        f_u=np.array(F_U),
        f_sampling=np.array(F_SAMPLING),
        f_ref=np.array(F_REF),
    )
    print(f"  Saved samples -> {npz_path}")

    summary = dict(
        waveform=waveform_name,
        n_samples=n_samples,
        T=T,
        seed=SEED,
        threshold=threshold,
        msa_fallback=int(msa_mask.sum()),
        mismatch=_summary_stats(mismatch, threshold),
        mismatch_hp=_summary_stats(mismatch_hp, threshold),
        mismatch_hc=_summary_stats(mismatch_hc, threshold),
    )
    json_path = out_dir / f"{waveform_name}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary -> {json_path}")
    s = summary["mismatch"]
    if s["min"] is not None:
        print(
            f"  mismatch  min={s['min']:.3e}  median={s['median']:.3e}  "
            f"mean={s['mean']:.3e}  max={s['max']:.3e}  "
            f"failed={s['n_failed']}/{s['n_finite']}  threshold={threshold:.0e}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--T", type=float, default=DEFAULT_T)
    parser.add_argument("--tag", type=str, required=True,
                        help="Run label (e.g. 'pre_fix', 'post_fix_t0').")
    parser.add_argument("--threshold", type=float, default=1e-6,
                        help="Mismatch threshold for pass/fail bookkeeping.")
    args = parser.parse_args()

    if not LAL_AVAILABLE:
        sys.exit("LALSuite is not available — install with `uv sync --group cross-validation`.")

    out_dir = REPO_ROOT / "baseline_results" / args.tag
    print(f"Output directory: {out_dir}")

    # Single shared parameter set (precessing format = 12 params), sub-sliced
    # for the aligned-spin XHM run so that the in-plane spins are simply
    # ignored.  This keeps both runs reproducible *and* coupled in mass/spin
    # magnitudes between the two waveforms — convenient for cross-comparison.
    theta_prec = generate_random_params(
        args.n_samples, BBH_BOUNDS, is_tidal=False, is_precessing=True, seed=SEED,
    )
    # XHM expects [m1, m2, s1z, s2z, dist, tc, phic, inc].
    # theta_prec columns: [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, dist, tc, phic, inc]
    theta_xhm = theta_prec[:, [0, 1, 4, 7, 8, 9, 10, 11]]

    summaries = {}
    summaries["IMRPhenomXHM"] = run_waveform(
        "IMRPhenomXHM", theta_xhm, args.n_samples, args.T, args.threshold, out_dir,
    )
    summaries["IMRPhenomXPHM"] = run_waveform(
        "IMRPhenomXPHM", theta_prec, args.n_samples, args.T, args.threshold, out_dir,
    )

    # ----- combined summary -----
    combined = {
        "tag": args.tag,
        "n_samples": args.n_samples,
        "T": args.T,
        "seed": SEED,
        "f_l": F_L, "f_u": F_U, "f_ref": F_REF,
        "psd": "ET_D_psd.txt",
        "waveforms": summaries,
    }
    combined_path = out_dir / "summary.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nWrote combined summary: {combined_path}")


if __name__ == "__main__":
    main()
