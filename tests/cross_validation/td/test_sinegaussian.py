"""Large-scale SineGaussian validation against LALSimulation.

This is a time-domain large-scale test, so it compares LAL and ripple on exactly the
same centered samples with an FFT-free normalized mismatch.  It also checks a
relative norm error: normalization makes overlap blind to a global amplitude-scale
regression.  See ``sinegaussian.py`` for the reference-grid convention.
"""

from pathlib import Path

import pytest

lalsim = pytest.importorskip("lalsimulation")

from tests.cross_validation.td.sinegaussian import (
    plot_results,
    run_test,
    write_results,
)

pytestmark = pytest.mark.accuracy

# Enforced limits for the direct time-domain comparison.
_MISMATCH_THRESHOLD = 1e-17
_NORM_ERROR_THRESHOLD = 1e-15


def test_sinegaussian_lalsimulation_large_scale(n_samples, accuracy_outdir, make_plots):
    """LAL's direct burst generator and ripple agree across a random sweep."""
    result = run_test(n_samples, lalsim=lalsim)
    outdir = Path(accuracy_outdir)
    results_file = write_results(result, outdir)
    print(f"\n  Results saved to: {results_file}")
    if make_plots:
        figure_file = plot_results(result, outdir)
        print(f"  Figure saved to: {figure_file}")

    if result.errors:
        pytest.fail(
            f"{len(result.errors)}/{n_samples} SineGaussian trials failed: "
            f"{list(result.errors.items())[:5]}"
        )
    assert result.valid_mask.any(), "No testable SineGaussian trials"
    assert result.max_mismatch < _MISMATCH_THRESHOLD, (
        f"max time-domain mismatch {result.max_mismatch:.2e} exceeds "
        f"{_MISMATCH_THRESHOLD:.0e}"
    )
    assert result.max_norm_error < _NORM_ERROR_THRESHOLD, (
        f"max relative norm error {result.max_norm_error:.2e} exceeds "
        f"{_NORM_ERROR_THRESHOLD:.0e}"
    )
