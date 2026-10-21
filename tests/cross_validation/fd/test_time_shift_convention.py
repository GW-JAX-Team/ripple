"""Check that the IMRPhenomPv2 models differ from the reference only by a time shift.

ripple places the waveform's time origin with the exact phase derivative, while LAL
estimates the same derivative from a cubic spline (see
docs/dev/reference_comparison_notes.md). The raw overlap loss is therefore dominated
by a microsecond-scale linear phase ramp. This test fits that time shift, removes it,
and requires the realigned waveforms to agree at the float64 floor, so any
discrepancy other than the time origin still fails.
"""

import jax.numpy as jnp
import pytest

import ripplegw
from ripplegw.constants import PI
from tests.cross_validation.runner import (
    default_grid,
    get_tolerance,
    load_psd,
    load_tolerances,
)
from tests.helpers.metrics import combined_overlap_loss, get_nyquist_mask
from tests.helpers.params import random_params_batch

_TOLERANCES = load_tolerances()

_PARAMS = [
    pytest.param(name, marks=pytest.mark.smoke)
    for name in ("IMRPhenomPv2", "IMRPhenomPv2_NRTidalv2")
]

# Upper bound on the time-origin difference (the spline error is tens of
# microseconds at most); it also keeps ``_fit_time_shift`` below its wrapping limit.
_MAX_TIME_SHIFT = 1e-4


def _fit_time_shift(h, h_ref, psd, frequencies):
    """Least-squares ``dt`` with ``h = h_ref * exp(-2 pi i f dt)`` in both polarizations.

    The phase difference stays below pi for ``|dt| < 1 / (2 f_max)``, so it needs no
    unwrapping.
    """
    cross = (h["p"] * jnp.conj(h_ref["p"]) + h["c"] * jnp.conj(h_ref["c"])) / psd
    weight = jnp.abs(cross)
    phase = jnp.angle(cross)
    return -jnp.sum(weight * frequencies * phase) / (
        2 * PI * jnp.sum(weight * frequencies**2)
    )


@pytest.mark.accuracy
@pytest.mark.parametrize("waveform_name", _PARAMS)
def test_reference_time_shift_only(
    waveform_name, reference, n_samples, segment_duration_override
):
    if not reference.supports(waveform_name):
        pytest.skip(f"{reference.name} backend does not support {waveform_name!r}")

    threshold = get_tolerance(
        _TOLERANCES, reference.name, waveform_name, "time_aligned_overlap_loss"
    )
    wf = ripplegw.waveform(waveform_name, f_ref=5.0)
    grid = default_grid(wf, T_override=segment_duration_override)
    params_batch = random_params_batch(wf, n_samples, seed=42)

    nyquist = get_nyquist_mask(grid.axis)
    psd_freqs, psd_vals = load_psd()
    psd = jnp.interp(grid.axis, jnp.asarray(psd_freqs), jnp.asarray(psd_vals))

    worst = 0.0
    for i in range(n_samples):
        params = {k: float(v[i]) for k, v in params_batch.items()}
        ref = reference.generate(waveform_name, params, grid)
        rip = wf(grid.axis, params)
        h_ref = {k: jnp.asarray(ref[k]) * nyquist for k in ("p", "c")}
        h = {k: jnp.asarray(rip[k]) * nyquist for k in ("p", "c")}

        raw = float(
            combined_overlap_loss(
                h["p"], h["c"], h_ref["p"], h_ref["c"], psd, grid.axis
            )
        )
        dt = float(_fit_time_shift(h, h_ref, psd, grid.axis))
        shift = jnp.exp(2j * PI * grid.axis * dt)
        aligned = float(
            combined_overlap_loss(
                h["p"] * shift, h["c"] * shift, h_ref["p"], h_ref["c"], psd, grid.axis
            )
        )
        print(
            f"\n  {waveform_name} sample {i}: raw loss {raw:.1e}, "
            f"time shift {dt * 1e6:+.3f} us, realigned loss {aligned:.1e}"
        )
        assert abs(dt) < _MAX_TIME_SHIFT, (
            f"{waveform_name} sample {i}: time-origin difference {dt:.2e} s exceeds "
            f"{_MAX_TIME_SHIFT:.0e} s against reference {reference.name!r}"
        )
        worst = max(worst, aligned)

    assert worst < threshold, (
        f"{waveform_name}: overlap loss after removing the time shift is {worst:.2e}, "
        f"above {threshold:.0e} -- the waveforms differ by more than their time origin "
        f"against reference {reference.name!r}"
    )
