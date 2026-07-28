"""Large-scale FFT-free validation of ``ExactPulsarSignal`` against LALPulsar.

Unlike the full CW models, the exact model intentionally excludes the
Einstein/Shapiro terms and therefore cannot use ``CWMakeFakeData`` as a
reference.  ``exact_runner.py`` instead constructs LAL's matching geometric
model directly from detector states and antenna coefficients over a randomized
parameter sweep.
"""

from pathlib import Path

import jax
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

_MISMATCH_THRESHOLD = 1e-9
_NORM_ERROR_THRESHOLD = 1e-8


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
    assert result.max_mismatch < _MISMATCH_THRESHOLD, (
        f"max time-domain mismatch {result.max_mismatch:.2e} exceeds "
        f"{_MISMATCH_THRESHOLD:.0e}"
    )
    assert result.max_relative_norm_error < _NORM_ERROR_THRESHOLD, (
        f"max relative norm error {result.max_relative_norm_error:.2e} exceeds "
        f"{_NORM_ERROR_THRESHOLD:.0e}"
    )
