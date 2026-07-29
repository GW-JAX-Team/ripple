"""Large-scale, direct-LAL validation for :class:`ExactPulsarSignal`.

``ExactPulsarSignal`` deliberately implements the geometric-only CW timing
model.  It therefore cannot be compared to ``CWMakeFakeData``: that high-level
LAL pipeline always includes the Einstein and Shapiro terms.  This adapter uses
the SWIG-exposed LALPulsar building blocks that implement the same *intended*
model instead:

* ``GetDetectorStates`` and ``ComputeAMCoeffs`` provide LAL's detector position
  and antenna response;
* their geometric ``n . rDetector`` delay supplies the source phase; and
* ripple's registered ``ExactPulsarSignal`` object is projected through the
  same LAL antenna response before comparison.

The comparison is directly between aligned detector-time samples.  Its white,
normalized mismatch and relative-norm diagnostic use no FFT, whitening, or
time/phase maximization.  A 512 Hz rate makes every sample both an exact GPS
nanosecond timestamp and exactly representable in LAL's REAL8 GPS conversion;
it covers the sampled 10--200 Hz band without adding a grid-induced phase floor.
"""

from __future__ import annotations

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

_START_GPS = 1_000_000_000
_SAMPLE_RATE = 512.0
_DURATION = 8.0
_N_SAMPLES_PER_TRIAL = int(_SAMPLE_RATE * _DURATION)
_F0_MIN, _F0_MAX = 10.0, 200.0
_F1_MIN, _F1_MAX = -1.0e-8, 0.0
_SITES = ("H1", "L1", "V1")
_RIPPLE_BATCH_SIZE = 32


@dataclass(frozen=True)
class Trial:
    """One physically valid isolated-CW parameter point."""

    index: int
    site: str
    alpha: float
    delta: float
    psi: float
    phi0: float
    aplus: float
    across: float
    f0: float
    f1: float


@dataclass
class LargeScaleTestResult:
    """Per-trial direct detector-strain comparison results."""

    n_samples: int
    trials: list[Trial]
    mismatches: np.ndarray
    relative_norm_errors: np.ndarray
    errors: dict[int, str] = field(default_factory=dict)

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.mismatches) & np.isfinite(self.relative_norm_errors)

    @property
    def testable(self) -> np.ndarray:
        return self.mismatches[self.valid_mask]

    @property
    def max_mismatch(self) -> float:
        values = self.testable
        return float(np.max(values)) if values.size else float("nan")

    @property
    def max_relative_norm_error(self) -> float:
        values = self.relative_norm_errors[self.valid_mask]
        return float(np.max(values)) if values.size else float("nan")


def sample_trials(n: int, seed: int = 42) -> list[Trial]:
    """Draw deterministic, physical isolated-CW trials.

    Frequency is log-uniform so the sweep retains coverage at both low and high
    phase accumulation.  ``aplus`` and ``across`` are derived from ``h0`` and
    ``cos(iota)``, enforcing the physical relation ``aplus >= abs(across)``.
    A single first spindown is exercised through the public waveform API.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(seed)
    trials = []
    for index in range(n):
        cos_iota = float(rng.uniform(-1.0, 1.0))
        h0 = float(rng.uniform(0.5, 1.5))
        trials.append(
            Trial(
                index=index,
                site=_SITES[index % len(_SITES)],
                alpha=float(rng.uniform(0.0, 2.0 * math.pi)),
                delta=float(math.asin(rng.uniform(-1.0, 1.0))),
                psi=float(rng.uniform(0.0, math.pi)),
                phi0=float(rng.uniform(0.0, 2.0 * math.pi)),
                aplus=h0 * (1.0 + cos_iota**2) / 2.0,
                across=h0 * cos_iota,
                f0=float(np.exp(rng.uniform(math.log(_F0_MIN), math.log(_F0_MAX)))),
                f1=float(rng.uniform(_F1_MIN, _F1_MAX)),
            )
        )
    return trials


def relative_time_axis() -> np.ndarray:
    """The exact sample offsets for the fixed 512 Hz large-scale-test grid."""
    return np.arange(_N_SAMPLES_PER_TRIAL, dtype=np.float64) / _SAMPLE_RATE


def _site_detector(lal, site: str):
    """Return LAL's cached detector for one supported site."""
    indices = {
        "H1": lal.LALDetectorIndexLHODIFF,
        "L1": lal.LALDetectorIndexLLODIFF,
        "V1": lal.LALDetectorIndexVIRGODIFF,
    }
    return lal.CachedDetectors[indices[site]]


def _timestamps(lal, lalpulsar):
    """Build exact integer-nanosecond timestamps for one large-scale-test trial."""
    timestamps = lalpulsar.CreateTimestampVector(_N_SAMPLES_PER_TRIAL)
    samples_per_second = int(_SAMPLE_RATE)
    nanoseconds_per_sample = 1_000_000_000 // samples_per_second
    for index in range(_N_SAMPLES_PER_TRIAL):
        second, sample = divmod(index, samples_per_second)
        timestamps.data[index] = lal.LIGOTimeGPS(
            _START_GPS + second, sample * nanoseconds_per_sample
        )
    timestamps.deltaT = 1.0 / _SAMPLE_RATE
    return timestamps


def _lal_reference_trial(trial: Trial, *, lal, lalpulsar, edat, timestamps):
    """Return LAL's geometric-model reference and its antenna response.

    This deliberately reproduces ``XLALSimulateExactPulsarSignal``'s REAL8 GPS
    arithmetic: each LIGO timestamp is converted with ``XLALGPSGetREAL8`` before
    subtracting the SSB reference time.  The large-scale-test grid is chosen so that this
    faithful reference convention does not itself introduce a sampling phase floor.
    """
    detector = _site_detector(lal, trial.site)
    states = lalpulsar.GetDetectorStates(timestamps, detector, edat, 0.0)
    sky_position = lal.SkyPosition()
    sky_position.system = lal.COORDINATESYSTEM_EQUATORIAL
    sky_position.longitude, sky_position.latitude = trial.alpha, trial.delta
    am_coefficients = lalpulsar.ComputeAMCoeffs(states, sky_position)
    a = np.asarray(am_coefficients.a.data, dtype=np.float64)
    b = np.asarray(am_coefficients.b.data, dtype=np.float64)
    detector_positions = np.asarray(
        [tuple(states.data[index].rDetector) for index in range(_N_SAMPLES_PER_TRIAL)],
        dtype=np.float64,
    )

    n_hat = np.array(
        [
            math.cos(trial.delta) * math.cos(trial.alpha),
            math.cos(trial.delta) * math.sin(trial.alpha),
            math.sin(trial.delta),
        ]
    )
    delays = detector_positions @ n_hat
    gps_times = np.asarray(
        [
            states.data[index].tGPS.gpsSeconds
            + 1.0e-9 * states.data[index].tGPS.gpsNanoSeconds
            for index in range(_N_SAMPLES_PER_TRIAL)
        ],
        dtype=np.float64,
    )
    start_time_ssb = gps_times[0] + delays[0]
    tau = (gps_times - start_time_ssb) + delays
    phase = 2.0 * math.pi * (trial.f0 * tau + 0.5 * trial.f1 * tau**2)

    arm_opening = abs(
        detector.frDetector.xArmAzimuthRadians - detector.frDetector.yArmAzimuthRadians
    )
    sin_zeta = math.sin(arm_opening)
    cos_2psi, sin_2psi = math.cos(2.0 * trial.psi), math.sin(2.0 * trial.psi)
    f_plus = sin_zeta * (a * cos_2psi + b * sin_2psi)
    f_cross = sin_zeta * (b * cos_2psi - a * sin_2psi)
    a_plus, a_cross = sin_zeta * trial.aplus, sin_zeta * trial.across
    a1 = (
        a_plus * math.cos(trial.phi0) * cos_2psi
        - a_cross * math.sin(trial.phi0) * sin_2psi
    )
    a2 = (
        a_plus * math.cos(trial.phi0) * sin_2psi
        + a_cross * math.sin(trial.phi0) * cos_2psi
    )
    a3 = (
        -a_plus * math.sin(trial.phi0) * cos_2psi
        - a_cross * math.cos(trial.phi0) * sin_2psi
    )
    a4 = (
        -a_plus * math.sin(trial.phi0) * sin_2psi
        + a_cross * math.cos(trial.phi0) * cos_2psi
    )
    strain = (
        a1 * a * np.cos(phase)
        + a2 * b * np.cos(phase)
        + a3 * a * np.sin(phase)
        + a4 * b * np.sin(phase)
    )
    return detector, f_plus, f_cross, strain


def _generate_lal_references(trials: list[Trial], *, lal, lalpulsar, edat):
    """Generate direct LAL references concurrently, preserving trial failures."""
    timestamps = _timestamps(lal, lalpulsar)

    def one(trial: Trial):
        try:
            detector, f_plus, f_cross, strain = _lal_reference_trial(
                trial,
                lal=lal,
                lalpulsar=lalpulsar,
                edat=edat,
                timestamps=timestamps,
            )
            return trial.index, detector, f_plus, f_cross, strain, None
        except Exception as exc:  # noqa: BLE001 - retain usable trials in a sweep
            return trial.index, None, None, None, None, str(exc)

    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cpus = os.cpu_count() or 1
    n_workers = max(1, min(len(trials), available_cpus))

    references = {}
    errors: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for index, detector, f_plus, f_cross, strain, error in pool.map(one, trials):
            if error is None:
                references[index] = (detector, f_plus, f_cross, strain)
            else:
                errors[index] = error
    return references, errors


def _ripple_parameters(trials: list[Trial], references) -> dict[str, jnp.ndarray]:
    """Pack public ``ExactPulsarSignal`` parameters for a vmapped evaluation."""
    locations = np.asarray(
        [tuple(references[trial.index][0].location) for trial in trials], dtype=float
    )
    return {
        "alpha": jnp.asarray([trial.alpha for trial in trials]),
        "delta": jnp.asarray([trial.delta for trial in trials]),
        "f0": jnp.asarray([trial.f0 for trial in trials]),
        "phi0": jnp.asarray([trial.phi0 for trial in trials]),
        "aplus": jnp.asarray([trial.aplus for trial in trials]),
        "across": jnp.asarray([trial.across for trial in trials]),
        "f1": jnp.asarray([trial.f1 for trial in trials]),
        "site_x": jnp.asarray(locations[:, 0]),
        "site_y": jnp.asarray(locations[:, 1]),
        "site_z": jnp.asarray(locations[:, 2]),
    }


def _generate_ripple_batch(waveform, axis: jnp.ndarray, trials, references):
    """Evaluate the registered waveform in bounded vmapped chunks."""
    evaluate_batch = jax.jit(jax.vmap(lambda params: waveform(axis, params)))
    plus, cross = [], []
    for start in range(0, len(trials), _RIPPLE_BATCH_SIZE):
        chunk = trials[start : start + _RIPPLE_BATCH_SIZE]
        output = evaluate_batch(_ripple_parameters(chunk, references))
        output["p"].block_until_ready()
        plus.append(np.asarray(output["p"]))
        cross.append(np.asarray(output["c"]))
    return np.concatenate(plus), np.concatenate(cross)


def run_large_scale_test(
    n_samples: int,
    *,
    lal,
    lalpulsar,
    earth_ephemeris_file: str,
    sun_ephemeris_file: str,
    seed: int = 42,
) -> LargeScaleTestResult:
    """Compare randomized ``ExactPulsarSignal`` trials to direct LAL building blocks."""
    trials = sample_trials(n_samples, seed=seed)
    result = LargeScaleTestResult(
        n_samples=n_samples,
        trials=trials,
        mismatches=np.full(n_samples, np.nan),
        relative_norm_errors=np.full(n_samples, np.nan),
    )

    edat = lalpulsar.InitBarycenter(earth_ephemeris_file, sun_ephemeris_file)
    references, errors = _generate_lal_references(
        trials, lal=lal, lalpulsar=lalpulsar, edat=edat
    )
    result.errors.update(errors)
    valid_trials = [trial for trial in trials if trial.index in references]
    if not valid_trials:
        return result

    waveform = ripplegw.waveform(
        "ExactPulsarSignal",
        earth_ephemeris_file=earth_ephemeris_file,
        sun_ephemeris_file=sun_ephemeris_file,
        start_gps=_START_GPS,
        n_spindowns=1,
    )
    ours_p, ours_c = _generate_ripple_batch(
        waveform, jnp.asarray(relative_time_axis()), valid_trials, references
    )
    for position, trial in enumerate(valid_trials):
        _detector, f_plus, f_cross, reference = references[trial.index]
        strain = f_plus * ours_p[position] + f_cross * ours_c[position]
        try:
            result.mismatches[trial.index] = time_domain_overlap_loss(strain, reference)
            result.relative_norm_errors[trial.index] = relative_norm_error(
                strain, reference
            )
        except ValueError as exc:
            result.errors[trial.index] = str(exc)
    return result


def _hardware_info() -> dict:
    return {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "sample_rate_hz": _SAMPLE_RATE,
        "duration_s": _DURATION,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # noqa: DTZ005 - display metadata
    }


def write_results(result: LargeScaleTestResult, outdir: Path) -> Path:
    """Write trial inputs, FFT-free diagnostics, and run metadata as JSON."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / f"exact_pulsar_lal_large_scale_test_n{result.n_samples}.json"
    payload = {
        "waveform": "ExactPulsarSignal",
        "reference": "LALPulsar detector states + AM coefficients",
        "n_samples": result.n_samples,
        "n_failed": len(result.errors),
        "max_time_domain_mismatch": result.max_mismatch,
        "max_relative_norm_error": result.max_relative_norm_error,
        "errors": result.errors,
        "trials": [
            {
                **asdict(trial),
                "time_domain_mismatch": float(result.mismatches[trial.index]),
                "relative_norm_error": float(result.relative_norm_errors[trial.index]),
            }
            for trial in result.trials
        ],
        "hardware": _hardware_info(),
    }
    out_file.write_text(json.dumps(payload, indent=2))
    return out_file


def plot_results(result: LargeScaleTestResult, outdir: Path) -> Path:
    """Plot mismatch and amplitude-error distributions without importing MPL by default."""
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    figure_file = outdir / f"exact_pulsar_lal_large_scale_test_n{result.n_samples}.png"
    valid = result.valid_mask
    mismatch = result.mismatches[valid]
    norm_error = result.relative_norm_errors[valid]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, values, title in (
        (axes[0], mismatch, r"$\log_{10}(m_{\mathrm{TD}})$"),
        (axes[1], norm_error, r"$\log_{10}(\mathrm{relative\ norm\ error})$"),
    ):
        positive = values[values > 0.0]
        if positive.size:
            axis.hist(np.log10(positive), bins=30, edgecolor="black", alpha=0.8)
        axis.set_xlabel(title)
        axis.set_ylabel("count")
    fig.suptitle(
        f"ExactPulsarSignal vs direct LAL building blocks (n={result.n_samples})"
    )
    fig.tight_layout()
    fig.savefig(str(figure_file), dpi=150)
    plt.close(fig)
    return figure_file
