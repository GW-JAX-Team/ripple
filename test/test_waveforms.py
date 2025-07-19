import numpy as np
import jax.numpy as jnp
import pytest

from ripplegw.waveforms.JaxNRSur import NRSurHyb3dq8_FD, NRSur7dq4_FD
from ripplegw.waveforms.WaveformModel import Polarization

@pytest.mark.parametrize("segment_length,sampling_rate", [
    (4.0, 2048),
    (8.0, 4096),
])
def test_nrsurhyb3dq8_fd_basic(segment_length, sampling_rate):
    model = NRSurHyb3dq8_FD(segment_length=segment_length, sampling_rate=sampling_rate)
    freqs = jnp.arange(20, 512, 1.0 / (segment_length))
    # M_tot, dist_mpc, q, chi_1z, chi_2z
    source_parameters = jnp.array([60.0, 500.0, 2.0, 0.5, -0.3])
    config_parameters = {}
    model_parameters = {}

    result = model.full_model(freqs, source_parameters, config_parameters, model_parameters)
    assert isinstance(result, dict)
    assert Polarization.P in result and Polarization.C in result
    assert result[Polarization.P].shape == freqs.shape
    assert result[Polarization.C].shape == freqs.shape

@pytest.mark.parametrize("segment_length,sampling_rate", [
    (4.0, 2048),
    (8.0, 4096),
])
def test_nrsur7dq4_fd_basic(segment_length, sampling_rate):
    model = NRSur7dq4_FD(segment_length=segment_length, sampling_rate=sampling_rate)
    freqs = jnp.arange(20, 512, 1.0 / (segment_length))
    # M_tot, dist_mpc, q, chi_1x, chi_1y, chi_1z, chi_2x, chi_2y, chi_2z
    source_parameters = jnp.array([60.0, 500.0, 1.5, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    config_parameters = {}
    model_parameters = {}

    result = model.full_model(freqs, source_parameters, config_parameters, model_parameters)
    assert isinstance(result, dict)
    assert Polarization.P in result and Polarization.C in result
    assert result[Polarization.P].shape == freqs.shape
    assert result[Polarization.C].shape == freqs.shape
