"""Shared support for frequency-domain cross-validation tests.

This module samples parameters, generates reference and ripple waveforms,
computes PSD-weighted overlap loss, and writes optional JSON results and plots.
"""

import json
import os
import platform
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import ripplegw
from tests.helpers.grids import Grid, accuracy_grid
from tests.helpers.metrics import combined_overlap_loss, get_nyquist_mask, overlap_loss
from tests.helpers.params import random_params_batch, regime

TOLERANCES_PATH = Path(__file__).parent / "tolerances.toml"
PSD_PATH = Path(__file__).parent.parent / "psds" / "ET_D_psd.txt"


def load_psd() -> tuple[np.ndarray, np.ndarray]:
    """Return the PSD used for frequency-domain overlap loss.

    The configured overlap-loss limits apply to this weighting.
    """
    freqs, psd = np.loadtxt(PSD_PATH, unpack=True)
    return freqs, psd


def load_tolerances(path: Path | None = None) -> dict:
    with open(path or TOLERANCES_PATH, "rb") as f:
        return tomllib.load(f)


def get_tolerance(tolerances: dict, backend: str, waveform: str, kind: str) -> float:
    """``kind`` is ``"overlap_loss"`` or ``"phase_offset"``.

    Falls back to ``[<backend>.defaults]`` per-key, not per-entry -- a model
    with only ``overlap_loss`` set still gets the default ``phase_offset``.
    """
    backend_table = tolerances.get(backend, {})
    entry = backend_table.get(waveform, {})
    if kind in entry:
        return entry[kind]
    return backend_table["defaults"][kind]


def default_grid(wf, *, T_override: float | None = None) -> Grid:
    """Return the regime-appropriate frequency grid for one waveform."""
    is_bns = regime(wf) == "bns"
    T = T_override if T_override is not None else (128.0 if is_bns else 32.0)
    f_u = 4096.0 if is_bns else 2048.0
    return accuracy_grid(f_l=5.0, f_u=f_u, f_sampling=2 * f_u, T=T, f_ref=5.0)


@dataclass
class TestResult:
    waveform: str
    reference: str
    n_samples: int
    grid: Grid
    overlap_loss_p: np.ndarray  # NaN for failed/excluded samples
    overlap_loss_c: np.ndarray
    overlap_loss_combined: np.ndarray  # SNR-weighted; this is the asserted metric
    valid_mask: np.ndarray
    expected_failures: np.ndarray  # excluded from both errors and valid_mask
    errors: dict = field(default_factory=dict)
    threshold: float = float("nan")

    @property
    def worst(self) -> np.ndarray:
        """Per-polarization worst case -- a diagnostic only.

        Amplified for the near-vanishing polarization at edge-on inclination;
        ``testable``/``max_loss`` use the SNR-weighted ``overlap_loss_combined``
        instead. See ``tests.helpers.metrics.combined_overlap_loss``.
        """
        return np.maximum(self.overlap_loss_p, self.overlap_loss_c)

    @property
    def testable(self) -> np.ndarray:
        c = self.overlap_loss_combined
        return c[self.valid_mask & np.isfinite(c)]

    @property
    def max_loss(self) -> float:
        t = self.testable
        return float(np.max(t)) if t.size else float("nan")

    @property
    def passed(self) -> bool:
        return (
            not self.errors
            and self.testable.size > 0
            and self.max_loss < self.threshold
        )


def _reference_cache_path(
    cache_dir: Path, backend: str, waveform: str, T: float, seed: int
) -> Path:
    T_str = f"T{int(T)}" if T == int(T) else f"T{T}"
    return cache_dir / f"{backend}_{waveform}_{T_str}_seed{seed}.npz"


def _load_reference_cache(path: Path, n_samples: int, n_freqs: int):
    if not path.exists():
        return None
    try:
        data = np.load(str(path))
    except Exception:  # noqa: BLE001 - any unreadable/corrupt cache file is just a cache miss
        return None
    if data["hp"].shape[0] < n_samples or data["hp"].shape[1] != n_freqs:
        return None
    out = {k: data[k][:n_samples] for k in ("hp", "hc", "valid")}
    out["expected_failures"] = (
        data["expected_failures"][:n_samples]
        if "expected_failures" in data.files
        else np.zeros(n_samples, dtype=bool)
    )
    return out


def _save_reference_cache(path: Path, hp, hc, valid, expected_failures, errors) -> None:
    if errors:
        # A cache that dropped an unexpected reference failure would let a later
        # cached run pass where the fresh run failed. Only cache clean batches.
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), hp=hp, hc=hc, valid=valid, expected_failures=expected_failures)


def generate_reference_batch(
    reference,
    waveform: str,
    params_batch: dict,
    grid: Grid,
    n_samples: int,
    *,
    cache_path: Path | None = None,
):
    """Reference-backend ``(hp, hc)`` for each sample, in parallel.

    LAL's C extension releases the GIL, so a thread pool gives real
    parallelism; other backends pay a small, harmless overhead. A sample the
    backend fails to generate is left NaN and excluded via ``valid``/``errors``.
    """
    if cache_path is not None:
        cached = _load_reference_cache(cache_path, n_samples, len(grid.axis))
        if cached is not None:
            return (
                cached["hp"],
                cached["hc"],
                cached["valid"].astype(bool),
                {},
                cached["expected_failures"].astype(bool),
            )

    is_expected_failure = getattr(reference, "expected_failure", None)

    def _one(i: int):
        p = {k: float(v[i]) for k, v in params_batch.items()}
        try:
            out = reference.generate(waveform, p, grid)
            return i, out["p"], out["c"], None, False
        except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the batch
            expected = bool(is_expected_failure and is_expected_failure(waveform, exc))
            return i, None, None, str(exc), expected

    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu = os.cpu_count() or 1
    n_workers = max(1, min(n_samples, n_cpu))

    n_freqs = len(grid.axis)
    hp = np.full((n_samples, n_freqs), np.nan, dtype=complex)
    hc = np.full((n_samples, n_freqs), np.nan, dtype=complex)
    valid = np.zeros(n_samples, dtype=bool)
    expected_failures = np.zeros(n_samples, dtype=bool)
    errors: dict = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, p_wave, c_wave, err, expected in pool.map(_one, range(n_samples)):
            if err is None:
                hp[i], hc[i] = p_wave, c_wave
                valid[i] = True
            elif expected:
                expected_failures[i] = True
            else:
                errors[i] = err

    if cache_path is not None:
        _save_reference_cache(cache_path, hp, hc, valid, expected_failures, errors)
    return hp, hc, valid, errors, expected_failures


def _is_oom(exc: Exception) -> bool:
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg


def generate_ripple_batch(wf, params_batch: dict, grid: Grid, valid_mask: np.ndarray):
    """ripple ``(hp, hc)`` for every valid sample, batched.

    Tries a full ``vmap`` first; on GPU OOM, falls back to ``jax.lax.map``
    with a shrinking ``batch_size``, and finally to a sequential Python loop
    (peak memory O(1 sample)) if even ``batch_size=1`` still OOMs.
    """
    idx = np.where(valid_mask)[0]
    n_valid = len(idx)
    params_dev = {k: jnp.asarray(np.asarray(v)[idx]) for k, v in params_batch.items()}

    def _call_one(p):
        out = wf(grid.axis, p)
        return out["p"], out["c"]

    batch_size = None  # None -> full vmap
    while True:
        try:
            if batch_size is None:
                fn = jax.jit(jax.vmap(_call_one))
            else:
                fn = jax.jit(
                    lambda p, bs=batch_size: jax.lax.map(_call_one, p, batch_size=bs)
                )
            hp, hc = fn(params_dev)
            hp.block_until_ready()
            return np.asarray(hp), np.asarray(hc), idx
        except Exception as exc:
            if not _is_oom(exc):
                raise
            next_batch = max(1, (n_valid if batch_size is None else batch_size) // 2)
            if batch_size is not None and next_batch == batch_size:
                single = jax.jit(_call_one)
                hp_list, hc_list = [], []
                for k in range(n_valid):
                    p_k = {key: v[k] for key, v in params_dev.items()}
                    hp_k, hc_k = single(p_k)
                    hp_list.append(np.asarray(hp_k))
                    hc_list.append(np.asarray(hc_k))
                return np.stack(hp_list), np.stack(hc_list), idx
            batch_size = next_batch


def run_overlap_test(
    name: str,
    reference,
    n_samples: int,
    threshold: float,
    *,
    T_override: float | None = None,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> TestResult:
    """End to end: sample parameters, generate both sides, score overlap loss."""
    wf = ripplegw.waveform(name, f_ref=5.0)
    grid = default_grid(wf, T_override=T_override)
    params_batch = random_params_batch(wf, n_samples, seed=seed)

    cache_path = (
        _reference_cache_path(cache_dir, reference.name, name, 1.0 / grid.df, seed)
        if cache_dir
        else None
    )
    ref_hp, ref_hc, valid, errors, expected_failures = generate_reference_batch(
        reference, name, params_batch, grid, n_samples, cache_path=cache_path
    )

    result = TestResult(
        waveform=name,
        reference=reference.name,
        n_samples=n_samples,
        grid=grid,
        overlap_loss_p=np.full(n_samples, np.nan),
        overlap_loss_c=np.full(n_samples, np.nan),
        overlap_loss_combined=np.full(n_samples, np.nan),
        valid_mask=valid,
        expected_failures=expected_failures,
        errors=errors,
        threshold=threshold,
    )
    if not valid.any():
        return result

    rip_hp, rip_hc, idx = generate_ripple_batch(wf, params_batch, grid, valid)
    nyquist = get_nyquist_mask(grid.axis)
    psd_freqs, psd_vals = load_psd()
    psd = jnp.interp(grid.axis, jnp.asarray(psd_freqs), jnp.asarray(psd_vals))

    for j, i in enumerate(idx):
        hp_r = jnp.asarray(rip_hp[j]) * nyquist
        hc_r = jnp.asarray(rip_hc[j]) * nyquist
        hp_ref = jnp.asarray(ref_hp[i]) * nyquist
        hc_ref = jnp.asarray(ref_hc[i]) * nyquist
        result.overlap_loss_p[i] = float(overlap_loss(hp_r, hp_ref, psd, grid.axis))
        result.overlap_loss_c[i] = float(overlap_loss(hc_r, hc_ref, psd, grid.axis))
        result.overlap_loss_combined[i] = float(
            combined_overlap_loss(hp_r, hc_r, hp_ref, hc_ref, psd, grid.axis)
        )

    return result


def _hardware_info() -> dict:
    devices = jax.devices()
    return {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "jax_devices": [str(d) for d in devices],
        "jax_version": jax.__version__,
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # noqa: DTZ005 - local display timestamp only
    }


def _run_tag(result: TestResult) -> str:
    T = round(1.0 / result.grid.df, 6)
    T_str = f"T{int(T)}" if T == int(T) else f"T{T}"
    return f"n{result.n_samples}_{T_str}"


def write_results(result: TestResult, outdir: Path) -> Path:
    """Persist per-sample overlap losses and run metadata as JSON under ``outdir``."""
    run_dir = outdir / _run_tag(result)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_file = run_dir / f"{result.reference}_{result.waveform}.json"
    payload = {
        "waveform": result.waveform,
        "reference": result.reference,
        "n_samples": result.n_samples,
        "threshold": result.threshold,
        "max_loss": result.max_loss,
        "passed": result.passed,
        "n_failed": len(result.errors),
        "errors": result.errors,
        "n_expected_failures": int(result.expected_failures.sum()),
        "overlap_loss_p": result.overlap_loss_p.tolist(),
        "overlap_loss_c": result.overlap_loss_c.tolist(),
        "overlap_loss_max": result.worst.tolist(),
        "overlap_loss_combined": result.overlap_loss_combined.tolist(),
        "hardware": _hardware_info(),
    }
    out_file.write_text(json.dumps(payload, indent=2))
    return out_file


def plot_results(result: TestResult, outdir: Path) -> Path:
    """Per-waveform overlap-loss histogram; only imports matplotlib when called."""
    import matplotlib.pyplot as plt

    fig_dir = outdir / _run_tag(result)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_file = fig_dir / f"{result.reference}_{result.waveform}.png"

    positive = result.testable[result.testable > 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    if positive.size:
        ax.hist(np.log10(positive), bins=30, edgecolor="black", alpha=0.8)
    ax.axvline(
        np.log10(result.threshold), color="red", linestyle="--", label="threshold"
    )
    ax.set_xlabel(r"$\log_{10}(1 - \mathcal{O})$")
    ax.set_ylabel("count")
    ax.set_title(f"{result.waveform} vs {result.reference}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fig_file), dpi=150)
    plt.close(fig)
    return fig_file
