"""Unit tests for FFT-free shared time-domain comparison metrics."""

import numpy as np
import pytest

from tests.cross_validation.cw._lal_helpers import overlap_loss as cw_overlap_loss
from tests.helpers.metrics import relative_norm_error, time_domain_overlap_loss


def test_time_domain_mismatch_is_zero_for_identical_series():
    h = np.array([0.2, -0.4, 0.1, 0.9])
    assert time_domain_overlap_loss(h, h) == pytest.approx(0.0)


def test_cw_overlap_loss_remains_a_backward_compatible_alias():
    h = np.array([0.2, -0.4, 0.1, 0.9])
    assert cw_overlap_loss(h, h) == pytest.approx(time_domain_overlap_loss(h, h))


def test_time_domain_mismatch_is_invariant_to_positive_amplitude_scale():
    h = np.array([0.2, -0.4, 0.1, 0.9])
    assert time_domain_overlap_loss(7.0 * h, h) == pytest.approx(0.0)


def test_time_domain_mismatch_handles_anti_correlated_series():
    h = np.array([0.2, -0.4, 0.1, 0.9])
    assert time_domain_overlap_loss(-h, h) == pytest.approx(2.0)


def test_relative_norm_error_detects_global_amplitude_scale_only():
    h = np.array([0.2, -0.4, 0.1, 0.9])
    assert relative_norm_error(1.01 * h, h) == pytest.approx(0.01)
    assert relative_norm_error(-h, h) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("h1", "h2", "message"),
    [
        (np.array([1.0]), np.array([1.0, 2.0]), "identical shapes"),
        (np.array([0.0, 0.0]), np.array([1.0, 2.0]), "zero strain"),
        (np.array([np.nan, 1.0]), np.array([1.0, 2.0]), "finite"),
        (np.array([1.0 + 1.0j]), np.array([1.0]), "real-valued"),
    ],
)
def test_time_domain_mismatch_rejects_invalid_inputs(h1, h2, message):
    with pytest.raises(ValueError, match=message):
        time_domain_overlap_loss(h1, h2)
