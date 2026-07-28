"""Large-scale FFT-free validation of ``ExactPulsarSignal`` against LALPulsar.

Unlike the full CW models, the exact model intentionally excludes the
Einstein/Shapiro terms and therefore cannot use ``CWMakeFakeData`` as a
reference.  ``exact_runner.py`` instead constructs LAL's matching geometric
model directly from detector states and antenna coefficients over a randomized
parameter sweep.

Like the ``CWMakeFakeData`` mismatch (see ``runner.py``), this mismatch is not a
flat floor either -- a 1000-trial run (2026-07-28, f0 in 10-200 Hz) found a clean
f0**2 scaling (log-log fit exponent ~1.97), just at a ~100x tighter absolute scale
(~1e-9 vs ~1e-6) since this is the exact-building-block methodology (LAL's own
REAL8 GPS-time phase evaluation, not the REAL4 CWMakeFakeData floor). The relative
norm error showed no comparable f0 trend (log-log correlation ~0.13).

Both thresholds below are "just above observed" (rounded up to the nearest power
of ten from the 2026-07-28 n=1000 run), not generously margined -- a fresh
large-scale test at a different ``--n-samples`` or parameter range may exceed
them and require re-deriving these constants.
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


# Pure f0**2 scaling, no additive floor needed -- the power law alone stays above
# every observed point in the 2026-07-28 n=1000 run (max 2.43e-9 at f0~191 Hz).
def _mismatch_threshold(f0: float) -> float:
    return 1e-9 * (f0 / 100.0) ** 2


# Flat: relative norm error showed no comparable f0 trend. Observed max 1.58e-8.
_NORM_ERROR_THRESHOLD = 1e-7


def test_exact_pulsar_signal_lal_large_scale(n_samples, accuracy_outdir, make_plots):
    """The registered geometric-only waveform agrees with LAL across a sweep."""
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
