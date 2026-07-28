"""Large-scale CW-vs-``CWMakeFakeData`` test: fits the mismatch-vs-f0 scaling law.

``test_makefakedata_v5.py`` checks agreement at one fixed point (f0=12.3 Hz), enough
for a small regression check but not enough to characterize the floor's actual
behavior. The direct, normalized time-domain mismatch vs. ``CWMakeFakeData`` is **not
a flat floor -- it scales as roughly f0**2** because LAL linearly interpolates a
barycentric-delay table with a hard-coded 400-second half interval (see
``runner.py``). This file runs the
comparison over a random draw of ``--n-samples`` trials spanning sky position,
amplitude parameters, f0 (log-uniform, 10-2000 Hz -- the typical CW all-sky search
band), spindown, detector site, and (for roughly half) binary orbital elements, then
fits ``loss(f0) ~= C * (f0/100Hz)**exponent`` per population. ``runner.py``'s
``mismatch_threshold(f0, is_binary)`` -- shared with ``test_makefakedata_v5.py``'s
single fixed point -- is calibrated from this fit (2026-07-28, n=1000 per
population), checked per trial against every sampled f0/population here, not just
against the reported max.

Not run by default CI (expensive at any useful ``--n-samples``). Launch a selected
CW model through the unified launcher, for example::

    python -m tests.cross_validation.submit --scheduler slurm \\
        --waveform PulsarSignal --n-samples 500 --outdir accuracy-results/cw --plots

Choose ``BinaryPulsarSignal`` instead to run its separate large-scale test.

Skipped, like every other file in this directory, unless both ``lalpulsar`` and an
Earth/Sun ephemeris file are available.
"""

from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

lal = pytest.importorskip("lal")
lalpulsar = pytest.importorskip("lalpulsar")

from tests.cross_validation.cw._lal_helpers import find_ephemeris
from tests.cross_validation.cw.runner import (
    MAX_RELATIVE_NORM_ERROR,
    mismatch_threshold,
    plot_results,
    run_large_scale_test,
    summarize,
    write_results,
)

EARTH_FILE, SUN_FILE = find_ephemeris()
pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.skipif(
        EARTH_FILE is None or SUN_FILE is None,
        reason="LALPulsar Earth and Sun ephemeris files required",
    ),
]

# mismatch_threshold and MAX_RELATIVE_NORM_ERROR live in runner.py, shared with
# test_makefakedata_v5.py's single fixed point -- see that module's docstring for
# the mechanism (f0**2 delay-table interpolation) and the antenna-response-table
# mechanism behind the flat norm-error ceiling.


def test_makefakedata_v5_large_scale(
    n_samples, accuracy_outdir, make_plots, cw_waveform
):
    result = run_large_scale_test(
        n_samples, lal=lal, lalpulsar=lalpulsar, waveform=cw_waveform
    )

    outdir = Path(accuracy_outdir)
    results_file = write_results(result, outdir)
    print(f"\n  Results saved to: {results_file}")
    if make_plots:
        fig_file = plot_results(result, outdir)
        print(f"  Figure saved to: {fig_file}")

    summary = summarize(result)
    label = cw_waveform or "mixed CW"
    print(f"\n  {label} vs CWMakeFakeData large-scale test (n={n_samples}):")
    for population in ("isolated", "binary", "overall"):
        s = summary[population]
        if s["n"] == 0:
            print(f"    {population:<10} n=0")
            continue
        fit = s["power_law_fit"]
        print(
            f"    {population:<10} n={s['n']:<6} mean={s['mean']:.2e} "
            f"median={s['median']:.2e} p99={s['p99']:.2e} max={s['max']:.2e}"
        )
        print(
            f"      fit: loss(f0) ~= {fit['coefficient_at_f0_ref']:.2e} * "
            f"(f0/{fit['f0_ref']:.0f}Hz)^{fit['exponent']:.2f}"
        )
    if result.errors:
        print(f"    {len(result.errors)}/{n_samples} trials failed to generate")
        for i, err in list(result.errors.items())[:5]:
            print(f"      trial {i}: {err}")

    if result.errors:
        pytest.fail(
            f"{len(result.errors)}/{n_samples} trials failed to generate: "
            f"{list(result.errors.items())[:5]}"
        )
    assert result.testable.size > 0, "No testable trials"

    worst_ratio, worst_trial, worst_loss = 0.0, None, 0.0
    for trial in result.trials:
        loss = result.losses[trial.index]
        if not np.isfinite(loss):
            continue
        ratio = loss / mismatch_threshold(trial.f0, is_binary=trial.is_binary)
        if ratio > worst_ratio:
            worst_ratio, worst_trial, worst_loss = ratio, trial, loss
    assert worst_ratio < 1.0, (
        f"time-domain mismatch {worst_loss:.2e} at f0={worst_trial.f0:.1f} Hz "
        f"(is_binary={worst_trial.is_binary}) exceeds frequency-scaled threshold "
        f"{mismatch_threshold(worst_trial.f0, is_binary=worst_trial.is_binary):.2e}"
    )

    assert result.max_relative_norm_error < MAX_RELATIVE_NORM_ERROR, (
        f"max relative norm error {result.max_relative_norm_error:.2e} exceeds "
        f"amplitude-scale ceiling {MAX_RELATIVE_NORM_ERROR:.2e}"
    )
