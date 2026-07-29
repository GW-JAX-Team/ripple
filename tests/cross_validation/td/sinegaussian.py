"""Large-scale, FFT-free SineGaussian cross-validation against LALSimulation.

Time-domain models do not share one universal reference API: the waveform-specific
adapter must choose its reference routine, parameter conversion, and aligned sample
grid.  They *do* share the direct sampled-time metrics in
``tests.helpers.metrics``.  This module is SineGaussian's adapter, using
``lalsimulation.SimBurstSineGaussian``.

LAL owns the output length and represents the epoch with integer GPS seconds plus
nanoseconds.  Reconstructing the axis from ``float(series.epoch)`` loses part of a
nanosecond and creates an artificial high-frequency phase error.  The reference
routine defines its samples exactly as ``(i - (n - 1) / 2) * delta_t``; we use that
same centered axis for ripple.
"""

import json
import math
import os
import platform
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import ripplegw
from tests.helpers.metrics import relative_norm_error, time_domain_overlap_loss

_F0_MIN, _F0_MAX = 30.0, 1024.0
_Q_MIN, _Q_MAX = 3.0, 40.0
_HRSS_MIN, _HRSS_MAX = 1e-24, 1e-20
_E_MIN, _E_MAX = 0.0, 0.95
_SAMPLE_RATE = 8192.0
_RIPPLE_BATCH_SIZE = 64


@dataclass(frozen=True)
class Trial:
    """One physically valid SineGaussian parameter point."""

    index: int
    Q: float
    f_0: float
    hrss: float
    phase: float
    e: float


@dataclass
class TestResult:
    """Per-polarization time-domain comparison results for one large-scale test."""

    n_samples: int
    trials: list[Trial]
    mismatch_p: np.ndarray
    mismatch_c: np.ndarray
    norm_error_p: np.ndarray
    norm_error_c: np.ndarray
    errors: dict[int, str] = field(default_factory=dict)

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.mismatch_p) & np.isfinite(self.mismatch_c)

    @property
    def worst_mismatch(self) -> np.ndarray:
        return np.maximum(self.mismatch_p, self.mismatch_c)

    @property
    def worst_norm_error(self) -> np.ndarray:
        return np.maximum(self.norm_error_p, self.norm_error_c)

    @property
    def max_mismatch(self) -> float:
        values = self.worst_mismatch[self.valid_mask]
        return float(np.max(values)) if values.size else float("nan")

    @property
    def max_norm_error(self) -> float:
        values = self.worst_norm_error[self.valid_mask]
        return float(np.max(values)) if values.size else float("nan")


def sample_trials(n: int, seed: int = 42) -> list[Trial]:
    """Draw ``n`` SineGaussian points over practical burst-search ranges.

    Frequency, quality factor, and ``hrss`` are sampled logarithmically so a
    test gives each decade comparable coverage.  ``e`` stops short of one:
    an exactly linearly polarized burst has identically zero cross strain, for
    which a normalized per-polarization overlap is undefined.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(seed)
    trials = []
    for index in range(n):
        trials.append(
            Trial(
                index=index,
                Q=float(np.exp(rng.uniform(math.log(_Q_MIN), math.log(_Q_MAX)))),
                f_0=float(np.exp(rng.uniform(math.log(_F0_MIN), math.log(_F0_MAX)))),
                hrss=float(
                    np.exp(rng.uniform(math.log(_HRSS_MIN), math.log(_HRSS_MAX)))
                ),
                phase=float(rng.uniform(0.0, 2.0 * math.pi)),
                e=float(rng.uniform(_E_MIN, _E_MAX)),
            )
        )
    return trials


def centered_time_axis(n_samples: int, delta_t: float) -> np.ndarray:
    """LAL's exact centered SineGaussian sample times, without epoch rounding."""
    if n_samples < 1 or n_samples % 2 == 0:
        raise ValueError("SineGaussian reference length must be a positive odd number")
    return (np.arange(n_samples, dtype=np.float64) - n_samples // 2) * delta_t


def _reference_trial(trial: Trial, lalsim):
    hp, hc = lalsim.SimBurstSineGaussian(
        trial.Q,
        trial.f_0,
        trial.hrss,
        trial.e,
        trial.phase,
        1.0 / _SAMPLE_RATE,
    )
    if hp.data.length != hc.data.length or hp.deltaT != hc.deltaT:
        raise RuntimeError("LAL returned inconsistent SineGaussian polarizations")
    return np.asarray(hp.data.data), np.asarray(hc.data.data), float(hp.deltaT)


def _generate_references(trials: list[Trial], lalsim):
    """Generate independent LAL references, retaining failures per trial."""

    def one(trial: Trial):
        try:
            hp, hc, delta_t = _reference_trial(trial, lalsim)
            return trial.index, hp, hc, delta_t, None
        except Exception as exc:  # noqa: BLE001 - preserve the rest of the test
            return trial.index, None, None, None, str(exc)

    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cpus = os.cpu_count() or 1
    n_workers = max(1, min(len(trials), available_cpus))

    reference: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    errors: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for index, hp, hc, delta_t, error in pool.map(one, trials):
            if error is None:
                reference[index] = (hp, hc, delta_t)
            else:
                errors[index] = error
    return reference, errors


def _generate_ripple_batch(trials: list[Trial], axis: np.ndarray):
    """Evaluate ripple in small vmapped chunks on one common centered axis."""
    waveform = ripplegw.waveform("SineGaussian")
    axis_device = jnp.asarray(axis)

    def call_one(params):
        output = waveform(axis_device, params)
        return output["p"], output["c"]

    call_batch = jax.jit(jax.vmap(call_one))
    plus, cross = [], []
    for start in range(0, len(trials), _RIPPLE_BATCH_SIZE):
        chunk = trials[start : start + _RIPPLE_BATCH_SIZE]
        params = {
            name: jnp.asarray([getattr(trial, name) for trial in chunk])
            for name in ("Q", "f_0", "hrss", "phase", "e")
        }
        hp, hc = call_batch(params)
        hp.block_until_ready()
        plus.append(np.asarray(hp))
        cross.append(np.asarray(hc))
    return np.concatenate(plus), np.concatenate(cross)


def run_test(n_samples: int, *, lalsim, seed: int = 42) -> TestResult:
    """Compare a randomized SineGaussian batch against LALSimulation.

    The reference generator returns a different (but always centered) length for
    each trial. Ripple is evaluated efficiently on a common axis and each output is
    sliced to its corresponding reference support before the direct time-domain
    comparison.
    """
    trials = sample_trials(n_samples, seed=seed)
    references, errors = _generate_references(trials, lalsim)
    result = TestResult(
        n_samples=n_samples,
        trials=trials,
        mismatch_p=np.full(n_samples, np.nan),
        mismatch_c=np.full(n_samples, np.nan),
        norm_error_p=np.full(n_samples, np.nan),
        norm_error_c=np.full(n_samples, np.nan),
        errors=errors,
    )
    if not references:
        return result

    delta_ts = {item[2] for item in references.values()}
    if len(delta_ts) != 1:
        raise RuntimeError("LAL returned multiple SineGaussian sample intervals")
    delta_t = delta_ts.pop()
    max_length = max(hp.size for hp, _, _ in references.values())
    axis = centered_time_axis(max_length, delta_t)

    valid_trials = [trial for trial in trials if trial.index in references]
    ripple_p, ripple_c = _generate_ripple_batch(valid_trials, axis)
    for position, trial in enumerate(valid_trials):
        reference_p, reference_c, _ = references[trial.index]
        start = (max_length - reference_p.size) // 2
        stop = start + reference_p.size
        ours_p = ripple_p[position, start:stop]
        ours_c = ripple_c[position, start:stop]
        result.mismatch_p[trial.index] = time_domain_overlap_loss(ours_p, reference_p)
        result.mismatch_c[trial.index] = time_domain_overlap_loss(ours_c, reference_c)
        result.norm_error_p[trial.index] = relative_norm_error(ours_p, reference_p)
        result.norm_error_c[trial.index] = relative_norm_error(ours_c, reference_c)
    return result


def _hardware_info() -> dict:
    return {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # noqa: DTZ005 - display timestamp
    }


def write_results(result: TestResult, outdir: Path) -> Path:
    """Write individual trials, summary values, and runtime metadata to JSON."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / f"sinegaussian_lalsimulation_n{result.n_samples}.json"
    payload = {
        "waveform": "SineGaussian",
        "reference": "lalsimulation.SimBurstSineGaussian",
        "n_samples": result.n_samples,
        "n_failed": len(result.errors),
        "max_mismatch": result.max_mismatch,
        "max_relative_norm_error": result.max_norm_error,
        "errors": result.errors,
        "trials": [
            {
                **asdict(trial),
                "time_domain_mismatch_p": float(result.mismatch_p[trial.index]),
                "time_domain_mismatch_c": float(result.mismatch_c[trial.index]),
                "relative_norm_error_p": float(result.norm_error_p[trial.index]),
                "relative_norm_error_c": float(result.norm_error_c[trial.index]),
            }
            for trial in result.trials
        ],
        "hardware": _hardware_info(),
    }
    out_file.write_text(json.dumps(payload, indent=2))
    return out_file


def plot_results(result: TestResult, outdir: Path) -> Path:
    """Plot distributions of the shape/phase and amplitude-scale diagnostics."""
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig_file = outdir / f"sinegaussian_lalsimulation_n{result.n_samples}.png"
    finite = result.valid_mask
    mismatch = result.worst_mismatch[finite]
    norm_error = result.worst_norm_error[finite]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, values, title in (
        (axes[0], mismatch, r"$\log_{10}(1 - \mathcal{O}_{\mathrm{TD}})$"),
        (axes[1], norm_error, r"$\log_{10}(\mathrm{relative\ norm\ error})$"),
    ):
        positive = values[values > 0]
        if positive.size:
            axis.hist(np.log10(positive), bins=30, edgecolor="black", alpha=0.8)
        axis.set_xlabel(title)
        axis.set_ylabel("count")
    fig.suptitle(f"SineGaussian vs LALSimulation (n={result.n_samples})")
    fig.tight_layout()
    fig.savefig(str(fig_file), dpi=150)
    plt.close(fig)
    return fig_file
