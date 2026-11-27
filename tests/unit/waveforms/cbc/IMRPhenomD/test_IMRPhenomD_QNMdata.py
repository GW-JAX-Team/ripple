"""Tests for ripplegw.waveforms.cbc.IMRPhenomD.IMRPhenomD_QNMdata."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ripplegw.waveforms.cbc.IMRPhenomD import IMRPhenomD_QNMdata


def test_qnm_grid_is_bit_identical_to_jax_linspace():
    """The QNM grid keeps the exact values ``jnp.linspace`` gave with x64 enabled.

    The splined QNM tables are evaluated on this grid, so any change of rounding in
    the grid moves IMRPhenomD-family waveforms at the 1e-15 level.
    """
    grid = IMRPhenomD_QNMdata.QNMData_a
    assert isinstance(grid, np.ndarray)
    assert grid.dtype == np.float64
    np.testing.assert_array_equal(grid, np.asarray(jnp.linspace(-1, 1, 500_000)))
