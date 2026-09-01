"""Compare reference-supported frequency-domain waveforms with a backend.

For every parameter draw, the test computes the PSD-weighted overlap loss of
both polarizations and requires the SNR-weighted combination
(``combined_overlap_loss``) to meet that waveform's limit; the per-polarization
values are kept as diagnostics. Every frequency-domain model is smoke-marked, so
the CI accuracy gate covers all of them; the heavier sweep (more samples, more
references) is the offline ``tests/cross_validation/submit.py`` path. Time-domain
models use their own cross-validation adapters.
"""

from pathlib import Path

import numpy as np
import pytest

import ripplegw
from tests.cross_validation.runner import (
    get_tolerance,
    load_tolerances,
    plot_results,
    run_overlap_test,
    write_results,
)

_TOLERANCES = load_tolerances()

_PARAMS = [
    pytest.param(name, marks=pytest.mark.smoke)
    for name in ripplegw.list_waveforms(domain="FD")
]


@pytest.mark.accuracy
@pytest.mark.parametrize("waveform_name", _PARAMS)
def test_reference_overlap(
    waveform_name,
    reference,
    n_samples,
    segment_duration_override,
    accuracy_outdir,
    cache_reference,
    make_plots,
    cross_val_results,
):
    if not reference.supports(waveform_name):
        pytest.skip(f"{reference.name} backend does not support {waveform_name!r}")

    threshold = get_tolerance(
        _TOLERANCES, reference.name, waveform_name, "overlap_loss"
    )
    outdir = Path(accuracy_outdir)
    result = run_overlap_test(
        waveform_name,
        reference,
        n_samples,
        threshold,
        T_override=segment_duration_override,
        cache_dir=(outdir / "cache") if cache_reference else None,
    )

    results_file = write_results(result, outdir)
    print(f"\n  Results saved to: {results_file}")
    if make_plots:
        fig_file = plot_results(result, outdir)
        print(f"  Figure saved to: {fig_file}")

    finite = result.testable
    n_expected_failures = int(result.expected_failures.sum())
    if n_expected_failures:
        print(
            f"\n  {n_expected_failures}/{n_samples} samples excluded: "
            f"{reference.name} reference reported an expected failure"
        )
    cross_val_results.append(
        {
            "waveform": waveform_name,
            "reference": reference.name,
            "n_samples": n_samples,
            "n_failed": len(result.errors),
            "n_expected_failures": n_expected_failures,
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "max": result.max_loss,
            "threshold": threshold,
            "passed": result.passed,
        }
    )

    if result.errors:
        sample_errors = list(result.errors.items())[:5]
        pytest.fail(
            f"{len(result.errors)}/{n_samples} samples failed to generate for "
            f"{waveform_name} against {reference.name}: {sample_errors}"
        )
    assert finite.size > 0, f"No testable samples for {waveform_name}"
    assert result.max_loss < threshold, (
        f"{waveform_name}: max overlap loss {result.max_loss:.2e} exceeds "
        f"threshold {threshold:.2e} for reference {reference.name!r}"
    )
