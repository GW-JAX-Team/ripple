"""Dependency-free guards for ExactPulsarSignal large-scale-test trial construction."""

import numpy as np
import pytest

from tests.cross_validation.cw.exact_runner import (
    _DURATION,
    _F0_MAX,
    _F0_MIN,
    _N_SAMPLES_PER_TRIAL,
    _SAMPLE_RATE,
    _SITES,
    relative_time_axis,
    sample_trials,
)


def test_sample_trials_are_deterministic_and_physical():
    first = sample_trials(9, seed=7)
    second = sample_trials(9, seed=7)

    assert first == second
    assert [trial.site for trial in first] == list(_SITES) * 3
    assert all(_F0_MIN <= trial.f0 <= _F0_MAX for trial in first)
    assert all(trial.aplus >= abs(trial.across) for trial in first)
    assert all(-np.pi / 2 <= trial.delta <= np.pi / 2 for trial in first)


def test_sample_trials_rejects_nonpositive_count():
    with pytest.raises(ValueError, match="positive"):
        sample_trials(0)


def test_relative_time_axis_is_exact_large_scale_test_grid():
    axis = relative_time_axis()

    assert axis.shape == (_N_SAMPLES_PER_TRIAL,)
    assert axis[0] == 0.0
    assert axis[-1] == _DURATION - 1.0 / _SAMPLE_RATE
    np.testing.assert_allclose(np.diff(axis), 1.0 / _SAMPLE_RATE)
