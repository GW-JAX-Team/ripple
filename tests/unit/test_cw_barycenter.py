"""Unit tests for CW barycentering geometry."""

import numpy as np

from ripplegw.waveforms.cw.barycenter import source_unit_vector


def test_source_unit_vector():
    """The source unit vector matches the analytic expression and is a unit."""
    alpha, delta = 1.1, -0.4
    n = np.asarray(source_unit_vector(alpha, delta))
    expected = np.array(
        [np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)]
    )
    np.testing.assert_allclose(n, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-14)
