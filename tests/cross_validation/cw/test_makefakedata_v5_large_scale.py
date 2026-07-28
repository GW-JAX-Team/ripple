"""Large-scale end-to-end CW validation against ``CWMakeFakeData``.

The test samples sky position, amplitudes, frequency, spindown, detector site,
duration, and binary orbital elements where applicable. Each testable trial
must meet the shared frequency-scaled mismatch limit and relative norm-error
limit. Output includes a diagnostic mismatch-frequency fit.

``--cw-waveform`` selects one model; direct pytest uses a mixed isolated/binary
sweep when it is omitted. The test requires ``lalpulsar`` and Earth/Sun
ephemerides.
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


def test_makefakedata_v5_large_scale(
    n_samples, accuracy_outdir, make_plots, cw_waveform
):
    """Validate the selected CW model across randomized LALPulsar trials."""
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
