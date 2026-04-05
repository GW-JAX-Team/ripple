"""Shared pytest fixtures for ripple tests.

This module provides fixtures for common test setup, including default
frequency grid parameters.
"""

import pytest

from tests.utils import get_freqs


@pytest.fixture
def default_freq_params():
    """Default frequency grid parameters for testing.

    Returns:
        Dictionary with keys: f_l, f_u, f_sampling, T, f_ref.
    """
    return {
        "f_l": 20.0,
        "f_u": 1024.0,
        "f_sampling": 2048.0,
        "T": 16.0,
        "f_ref": 20.0,
    }


@pytest.fixture
def freq_grid(default_freq_params):
    """Frequency grid constructed from default parameters.

    Returns:
        Frequency array (jax.numpy.ndarray).
    """
    return get_freqs(
        default_freq_params["f_l"],
        default_freq_params["f_u"],
        default_freq_params["f_sampling"],
        default_freq_params["T"],
    )
