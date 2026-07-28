"""Batch machinery for the CW-vs-``CWMakeFakeData`` large-scale accuracy campaign.

Not a test module: this is what ``test_makefakedata_v5_campaign.py`` needs beyond
the assertion itself. ``test_makefakedata_v5.py`` checks ``PulsarSignal``/
``BinaryPulsarSignal`` against ``XLALCWMakeFakeData`` at a single, fixed point in
parameter space (f0=12.3 Hz) -- enough to catch regressions cheaply in CI, but not
enough to characterize the floor's actual behavior. This module runs the same
comparison over a random Monte Carlo sweep (sky position, amplitude parameters,
frequency, spindown, detector site, binary orbital elements) and fits how the
overlap loss scales with ``f0``, so the threshold can be set from an observed
*distribution and trend* instead of one point. Reuses ``_lal_helpers.py``'s
``make_fake_data_v5``/``detector_strain_from_am_response``/``overlap_loss`` --
same methodology, more coverage.

**The overlap loss is not a flat ~1e-7 floor -- it scales as roughly f0**2**,
confirmed both analytically (traced through ``CWMakeFakeData.c`` ->
``XLALGenerateCWSignalTS`` -> ``XLALGeneratePulsarSignal`` ->
``XLALPulsarSimulateCoherentGW`` in the local LALSuite checkout) and by four
targeted differential experiments:

- Band/sampling-rate invariant (10/31/100 Hz bands at fixed f0: identical loss).
- Duration invariant (16s-3600s at fixed f0: flat/slightly decreasing, not growing).
- ``sourceDeltaT``-invariant (60s down to 0.01s source-table cadence: zero effect,
  ruling out phase-table linear-interpolation error).
- **f0-driven, not heterodyne-driven**: decoupling the heterodyne frequency from f0
  (fHet=495 vs fHet=1, both at f0=500) gives identical loss (1.360e-4 vs 1.359e-4).
  This proves the error tracks the pulsar's absolute f0, not the residual/heterodyned
  frequency, i.e. it's baked into the source phase-table computation itself.
- Ruled out as a ripple-side effect: casting ripple's own float64 output through a
  float32 round-trip gives ~1e-16 loss with zero f0-dependence -- 12 orders of
  magnitude too small and flat, so this is not "ripple needs more precision".

The mechanism is consistent with what ``docs/dev/reference_implementations.md``
already documents for the building-block tests -- LAL evaluating phase from REAL8
GPS-second-scale times (~1e9) without ripple's own int+frac split -- just far more
prominent in this code path (``XLALGenerateSpinOrbitCW``'s source phase table,
consumed via a ``REAL4TimeSeries`` pipeline throughout ``CWMakeFakeData.c``) than in
the direct-``XLALBarycenter`` building-block tests' ~1e-9 to ~1e-13.

Duration is *not* swept past 1hr here: confirmed duration-independent (see above);
this campaign's value is breadth across sky position/frequency/site/orbital
parameters, not pushing duration further.
"""

import json
import math
import os
import platform
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ripplegw.waveforms.cw.ephemeris import read_ephemeris_file
from ripplegw.waveforms.cw.pulsar_signal import (
    generate_binary_pulsar_polarizations,
    generate_pulsar_polarizations,
)
from tests.cross_validation.cw._lal_helpers import (
    detector_strain_from_am_response,
    find_ephemeris,
    make_fake_data_v5,
    overlap_loss,
)

_SITES = ("H1", "L1", "V1")
_DURATIONS = (16.0, 100.0, 1000.0, 3600.0)  # seconds -- see module docstring
_START_GPS = 1_000_000_000


@dataclass
class Trial:
    index: int
    is_binary: bool
    site: str
    duration: float
    alpha: float
    delta: float
    psi: float
    phi0: float
    aplus: float
    across: float
    f0: float
    f1: float
    asini: float = 0.0
    ecc: float = 0.0
    period: float = 0.0
    argp: float = 0.0
    tp_ssb_offset: float = 0.0


_F0_MIN, _F0_MAX = 10.0, 2000.0  # spans the typical CW all-sky search band


def sample_trials(n: int, seed: int = 42) -> list[Trial]:
    """Random draw of ``n`` trials spanning sky position, amplitude parameters
    (via ``cos(iota)``/``h0`` so ``aplus >= |across|`` always holds), frequency,
    spindown, detector site, duration, and -- for roughly half the trials --
    binary orbital elements.

    ``f0`` is drawn log-uniform over ``[_F0_MIN, _F0_MAX]`` (10-2000 Hz, the
    typical CW all-sky search band) rather than linear-uniform: the overlap loss
    follows a power law in f0 (see module docstring), and log-uniform sampling
    gives even leverage across decades for fitting that law's exponent.
    """
    rng = np.random.default_rng(seed)
    trials = []
    for i in range(n):
        alpha = float(rng.uniform(0.0, 2 * math.pi))
        delta = float(math.asin(rng.uniform(-1.0, 1.0)))
        psi = float(rng.uniform(0.0, math.pi))
        phi0 = float(rng.uniform(0.0, 2 * math.pi))
        cos_iota = float(rng.uniform(-1.0, 1.0))
        h0 = float(rng.uniform(0.5, 1.5))
        aplus = h0 * (1.0 + cos_iota**2) / 2.0
        across = h0 * cos_iota
        f0 = float(np.exp(rng.uniform(math.log(_F0_MIN), math.log(_F0_MAX))))
        f1 = float(rng.uniform(-1e-8, 0.0))
        site = _SITES[i % len(_SITES)]
        duration = _DURATIONS[i % len(_DURATIONS)]
        is_binary = bool(i % 2)

        kwargs = {
            "index": i,
            "is_binary": is_binary,
            "site": site,
            "duration": duration,
            "alpha": alpha,
            "delta": delta,
            "psi": psi,
            "phi0": phi0,
            "aplus": aplus,
            "across": across,
            "f0": f0,
            "f1": f1,
        }
        if is_binary:
            period = float(rng.uniform(3600.0, 10 * 3600.0))
            kwargs.update(
                asini=float(rng.uniform(0.5, 5.0)),
                ecc=float(rng.uniform(0.0, 0.3)),
                period=period,
                argp=float(rng.uniform(0.0, 2 * math.pi)),
                tp_ssb_offset=float(rng.uniform(0.0, period)),
            )
        trials.append(Trial(**kwargs))
    return trials


def run_trial(
    trial: Trial, *, lal, lalpulsar, edat, eph, seph
) -> tuple[float, str | None]:
    """Run one trial: ``CWMakeFakeData`` vs. ripple, return ``(overlap_loss, error)``.

    ``error`` is ``None`` on success, else a string and ``overlap_loss`` is ``nan``
    -- mirrors ``campaign.generate_reference_batch``'s convention of never letting
    one bad sample kill the whole sweep.
    """
    det_index = {
        "H1": lal.LALDetectorIndexLHODIFF,
        "L1": lal.LALDetectorIndexLLODIFF,
        "V1": lal.LALDetectorIndexVIRGODIFF,
    }[trial.site]
    det = lal.CachedDetectors[det_index]
    # Half-band must cover both Earth's own Doppler modulation (~1e-4 * f0) and, for
    # binary trials, the orbital radial-velocity swing f0*asini*(2*pi/period) -- at
    # high f0 with large asini/short period this can reach several Hz, wider than a
    # fixed 5 Hz margin (XLALCWSignalCoveringBand then rejects the requested band
    # with XLAL_EINVAL "Invalid argument", empirically ~1% of binary trials at f0
    # up to 2000 Hz without this).
    doppler = trial.f0 * 1.1e-4
    if trial.is_binary and trial.period > 0:
        doppler += (
            trial.f0 * trial.asini * (2 * math.pi / trial.period) * (1 + trial.ecc)
        )
    half_band = 5.0 + 2.0 * doppler
    fmin = max(0.1, trial.f0 - half_band)
    band = 2.0 * half_band

    try:
        common = {
            "lal": lal,
            "lalpulsar": lalpulsar,
            "edat": edat,
            "det_prefix": det.frDetector.prefix,
            "alpha": trial.alpha,
            "delta": trial.delta,
            "psi": trial.psi,
            "phi0": trial.phi0,
            "aplus": trial.aplus,
            "across": trial.across,
            "f0": trial.f0,
            "start_gps": _START_GPS,
            "duration": trial.duration,
            "fmin": fmin,
            "band": band,
        }
        if trial.is_binary:
            tp_ssb = _START_GPS + trial.tp_ssb_offset
            tseries = make_fake_data_v5(
                **common,
                asini=trial.asini,
                ecc=trial.ecc,
                period=trial.period,
                argp=trial.argp,
                tp_ssb=tp_ssb,
            )
        else:
            tseries = make_fake_data_v5(**common, fkdot=(trial.f1,))

        n = tseries.data.length
        t_rel = (float(tseries.epoch) - _START_GPS) + jnp.arange(
            n, dtype=jnp.float64
        ) * tseries.deltaT

        if trial.is_binary:
            hp, hc = generate_binary_pulsar_polarizations(
                t_rel,
                _START_GPS,
                trial.alpha,
                trial.delta,
                trial.f0,
                trial.phi0,
                trial.aplus,
                trial.across,
                trial.asini,
                trial.ecc,
                trial.period,
                trial.argp,
                tp_ssb,
                tuple(det.location),
                eph.gps0,
                eph.dt,
                jnp.asarray(eph.pos),
                jnp.asarray(eph.vel),
                jnp.asarray(eph.acc),
                seph.gps0,
                seph.dt,
                jnp.asarray(seph.pos),
                jnp.asarray(seph.vel),
                jnp.asarray(seph.acc),
                fkdot=(),
                ref_time_ssb=float(_START_GPS),
                f_heterodyne=float(tseries.f0),
            )
        else:
            hp, hc = generate_pulsar_polarizations(
                t_rel,
                _START_GPS,
                trial.alpha,
                trial.delta,
                trial.f0,
                trial.phi0,
                trial.aplus,
                trial.across,
                tuple(det.location),
                eph.gps0,
                eph.dt,
                jnp.asarray(eph.pos),
                jnp.asarray(eph.vel),
                jnp.asarray(eph.acc),
                seph.gps0,
                seph.dt,
                jnp.asarray(seph.pos),
                jnp.asarray(seph.vel),
                jnp.asarray(seph.acc),
                fkdot=(trial.f1,),
                ref_time_ssb=float(_START_GPS),
                f_heterodyne=float(tseries.f0),
            )

        h_mine = detector_strain_from_am_response(
            lal, det, trial.alpha, trial.delta, trial.psi, tseries, hp, hc
        )
        h_ref = tseries.data.data
        return overlap_loss(h_mine, h_ref), None
    except Exception as exc:  # noqa: BLE001 - one bad trial must not kill the sweep
        return float("nan"), str(exc)


@dataclass
class CampaignResult:
    n_samples: int
    trials: list[Trial]
    losses: np.ndarray
    errors: dict = field(default_factory=dict)

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.losses)

    @property
    def testable(self) -> np.ndarray:
        return self.losses[self.valid_mask]

    @property
    def max_loss(self) -> float:
        t = self.testable
        return float(np.max(t)) if t.size else float("nan")


def run_campaign(n_samples: int, *, lal, lalpulsar, seed: int = 42) -> CampaignResult:
    """Sample ``n_samples`` trials and run them in parallel.

    Uses a thread pool (not a process pool): LAL's C extension releases the GIL
    during ``CWMakeFakeData``/``ComputeDetAMResponse``, so threads give real
    parallelism here without the pickling overhead of ``multiprocessing`` --
    same rationale as ``cross_validation/campaign.py::generate_reference_batch``.
    """
    earth_file, sun_file = find_ephemeris()
    if earth_file is None or sun_file is None:
        raise RuntimeError("LALPulsar Earth/Sun ephemeris files not found")

    edat = lalpulsar.InitBarycenter(earth_file, sun_file)
    eph = read_ephemeris_file(earth_file)
    seph = read_ephemeris_file(sun_file)

    trials = sample_trials(n_samples, seed=seed)

    def _one(trial: Trial):
        return trial.index, *run_trial(
            trial, lal=lal, lalpulsar=lalpulsar, edat=edat, eph=eph, seph=seph
        )

    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu = os.cpu_count() or 1
    n_workers = max(1, min(n_samples, n_cpu))

    losses = np.full(n_samples, np.nan)
    errors: dict = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for i, loss, err in pool.map(_one, trials):
            losses[i] = loss
            if err is not None:
                errors[i] = err

    return CampaignResult(
        n_samples=n_samples, trials=trials, losses=losses, errors=errors
    )


def _hardware_info() -> dict:
    import jax

    return {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "n_cpu": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1),
        "python": platform.python_version(),
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # noqa: DTZ005 - local display timestamp only
    }


_F0_REF = 100.0  # Hz -- reference frequency for reporting the fitted coefficient


def fit_power_law(f0: np.ndarray, loss: np.ndarray) -> dict:
    """Least-squares fit of ``log(loss) = log(A) + exponent * log(f0)`` over positive,
    finite losses.

    Reported as ``loss(f0) ~= coefficient_at_f0_ref * (f0 / _F0_REF)**exponent`` --
    the raw intercept (``A`` at ``f0=1``) is numerically awkward to interpret since
    ``f0=1`` is far outside the sampled 10-2000 Hz range; anchoring at ``_F0_REF``
    keeps the reported coefficient at a representative magnitude.
    """
    mask = np.isfinite(loss) & (loss > 0)
    if mask.sum() < 3:
        return {
            "n": int(mask.sum()),
            "exponent": float("nan"),
            "coefficient_at_f0_ref": float("nan"),
            "f0_ref": _F0_REF,
        }
    log_f0 = np.log(f0[mask])
    log_loss = np.log(loss[mask])
    exponent, intercept = np.polyfit(log_f0, log_loss, 1)
    coefficient_at_f0_ref = float(np.exp(intercept + exponent * math.log(_F0_REF)))
    return {
        "n": int(mask.sum()),
        "exponent": float(exponent),
        "coefficient_at_f0_ref": coefficient_at_f0_ref,
        "f0_ref": _F0_REF,
    }


def summarize(result: CampaignResult) -> dict:
    """Per-population (isolated/binary/overall) percentile summary plus a fitted
    ``loss ~= C * (f0/f0_ref)**exponent`` power law (see module docstring -- the
    overlap loss scales with f0, not a flat floor)."""
    f0 = np.array([t.f0 for t in result.trials])

    def _stats(mask: np.ndarray) -> dict:
        sel = mask & result.valid_mask
        vals = result.losses[sel]
        if vals.size == 0:
            return {"n": 0}
        return {
            "n": int(vals.size),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p99": float(np.percentile(vals, 99)),
            "max": float(vals.max()),
            "power_law_fit": fit_power_law(f0[sel], vals),
        }

    is_binary = np.array([t.is_binary for t in result.trials])
    return {
        "isolated": _stats(~is_binary),
        "binary": _stats(is_binary),
        "overall": _stats(np.ones_like(is_binary)),
        "n_failed": len(result.errors),
    }


def write_results(result: CampaignResult, outdir: Path) -> Path:
    """Persist per-trial parameters/losses, the summary, and run metadata as JSON."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / f"cw_makefakedata_v5_campaign_n{result.n_samples}.json"
    payload = {
        "n_samples": result.n_samples,
        "n_failed": len(result.errors),
        "errors": result.errors,
        "summary": summarize(result),
        "trials": [
            {**asdict(t), "overlap_loss": result.losses[t.index]} for t in result.trials
        ],
        "hardware": _hardware_info(),
    }
    out_file.write_text(json.dumps(payload, indent=2))
    return out_file


def plot_results(result: CampaignResult, outdir: Path) -> Path:
    """Log-log overlap-loss vs. f0 scatter (isolated vs. binary) with the fitted
    power law overlaid -- the informative view here is the f0-scaling itself, not
    just the loss distribution (see module docstring). Only imports matplotlib
    when called."""
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig_file = outdir / f"cw_makefakedata_v5_campaign_n{result.n_samples}.png"

    f0 = np.array([t.f0 for t in result.trials])
    is_binary = np.array([t.is_binary for t in result.trials])
    valid = result.valid_mask
    summary = summarize(result)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for mask, label, color in (
        (~is_binary, "isolated", "tab:blue"),
        (is_binary, "binary", "tab:orange"),
    ):
        sel = mask & valid
        f0_pop, loss_pop = f0[sel], result.losses[sel]
        keep = loss_pop > 0
        if keep.any():
            ax.scatter(
                f0_pop[keep], loss_pop[keep], s=12, alpha=0.6, label=label, color=color
            )
        fit = summary[label]["power_law_fit"]
        if np.isfinite(fit["exponent"]):
            f0_line = np.geomspace(_F0_MIN, _F0_MAX, 100)
            loss_line = (
                fit["coefficient_at_f0_ref"]
                * (f0_line / fit["f0_ref"]) ** fit["exponent"]
            )
            ax.plot(
                f0_line,
                loss_line,
                "--",
                color=color,
                label=f"{label} fit (exp={fit['exponent']:.2f})",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("f0 (Hz)")
    ax.set_ylabel(r"overlap loss $1 - \mathcal{O}$")
    ax.set_title(f"CW vs CWMakeFakeData: loss vs f0 (n={result.n_samples})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(fig_file), dpi=150)
    plt.close(fig)
    return fig_file
