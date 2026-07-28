"""Large-scale direct-LAL validation of ``ExactPulsarSignal``.

``ExactPulsarSignal`` deliberately uses geometric-only timing, so the test
builds LAL's matching reference from detector states and antenna coefficients
rather than ``CWMakeFakeData``. Each randomized trial is checked with a
frequency-scaled time-domain mismatch limit and a relative norm-error limit.
"""

from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

lal = pytest.importorskip("lal")
lalpulsar = pytest.importorskip("lalpulsar")

from tests.cross_validation.cw._lal_helpers import find_ephemeris
from tests.cross_validation.cw.exact_runner import (
    plot_results,
    run_large_scale_test,
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


# Frequency-scaled mismatch limit for the direct-LAL comparison.
def _mismatch_threshold(f0: float) -> float:
    return 1e-9 * (f0 / 100.0) ** 2


# Relative amplitude-scale limit for the direct-LAL comparison.
_NORM_ERROR_THRESHOLD = 1e-7


def test_exact_pulsar_signal_lal_large_scale(n_samples, accuracy_outdir, make_plots):
    """Validate the registered geometric-only waveform across a randomized sweep."""
    result = run_large_scale_test(
        n_samples,
        lal=lal,
        lalpulsar=lalpulsar,
        earth_ephemeris_file=EARTH_FILE,
        sun_ephemeris_file=SUN_FILE,
    )
    outdir = Path(accuracy_outdir)
    results_file = write_results(result, outdir)
    print(f"\n  Results saved to: {results_file}")
    if make_plots:
        figure_file = plot_results(result, outdir)
        print(f"  Figure saved to: {figure_file}")

    if result.errors:
        pytest.fail(
            f"{len(result.errors)}/{n_samples} ExactPulsarSignal trials failed: "
            f"{list(result.errors.items())[:5]}"
        )
    assert result.valid_mask.any(), "No testable ExactPulsarSignal trials"

    worst_ratio, worst_trial, worst_mismatch = 0.0, None, 0.0
    for trial in result.trials:
        mismatch = result.mismatches[trial.index]
        if not np.isfinite(mismatch):
            continue
        ratio = mismatch / _mismatch_threshold(trial.f0)
        if ratio > worst_ratio:
            worst_ratio, worst_trial, worst_mismatch = ratio, trial, mismatch
    assert worst_ratio < 1.0, (
        f"time-domain mismatch {worst_mismatch:.2e} at f0={worst_trial.f0:.1f} Hz "
        f"exceeds frequency-scaled threshold {_mismatch_threshold(worst_trial.f0):.2e}"
    )

    assert result.max_relative_norm_error < _NORM_ERROR_THRESHOLD, (
        f"max relative norm error {result.max_relative_norm_error:.2e} exceeds "
        f"{_NORM_ERROR_THRESHOLD:.0e}"
    )
