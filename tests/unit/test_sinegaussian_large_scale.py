"""Pure checks for the SineGaussian time-domain large-scale test setup."""

import numpy as np
import pytest

from tests.cross_validation.td.sinegaussian import centered_time_axis, sample_trials


def test_centered_time_axis_matches_lal_sample_convention():
    axis = centered_time_axis(5, 0.25)
    np.testing.assert_array_equal(axis, np.array([-0.5, -0.25, 0.0, 0.25, 0.5]))


@pytest.mark.parametrize("length", [0, 2, 4])
def test_centered_time_axis_requires_an_odd_positive_reference_length(length):
    with pytest.raises(ValueError, match="positive odd"):
        centered_time_axis(length, 0.25)


def test_sinegaussian_large_scale_sampling_is_deterministic_and_physical():
    first = sample_trials(4, seed=123)
    second = sample_trials(4, seed=123)
    assert first == second
    assert [trial.index for trial in first] == list(range(4))
    for trial in first:
        assert trial.Q > 0.0
        assert trial.f_0 > 0.0
        assert trial.hrss > 0.0
        assert 0.0 <= trial.e < 1.0
