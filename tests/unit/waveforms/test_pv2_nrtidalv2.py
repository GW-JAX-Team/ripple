"""Unit checks for IMRPhenomPv2_NRTidalv2 that do not need a LAL reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import ripplegw
from ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomPv2_NRTidalv2 import (
    _natural_cubic_spline_derivative,
)
from tests.helpers.params import canonical_params

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("x_eval", [1.0, 1.13, 1.2, 0.81])
def test_spline_derivative_matches_scipy_natural_spline(x_eval):
    x = np.linspace(0.8, 1.2, 10)
    y = np.sin(7.0 * x) + 0.3 * x**2
    expected = CubicSpline(x, y, bc_type="natural")(x_eval, 1)
    got = _natural_cubic_spline_derivative(jnp.asarray(x), jnp.asarray(y), x_eval)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_reduces_to_aligned_waveform_amplitude_without_in_plane_spin():
    """With zero in-plane spin the precession angles vanish, so |h| must equal the
    IMRPhenomD_NRTidalv2 amplitude up to the inclination factors (PhenomP swaps
    body labels, so only the amplitude is compared)."""
    f = jnp.linspace(25.0, 900.0, 400)
    prec = ripplegw.waveform("IMRPhenomPv2_NRTidalv2", f_ref=20.0)
    aligned = ripplegw.waveform("IMRPhenomD_NRTidalv2", f_ref=20.0)
    p = canonical_params(prec, s1_x=0.0, s1_y=0.0, s2_x=0.0, s2_y=0.0, iota=0.7)
    pa = canonical_params(aligned, iota=0.7, s1_z=p["s1_z"], s2_z=p["s2_z"])
    hp, ha = prec(f, p)["p"], aligned(f, pa)["p"]
    np.testing.assert_allclose(np.abs(hp), np.abs(ha), rtol=1e-6)


def test_no_taper_only_changes_amplitude_above_merger():
    f = jnp.linspace(25.0, 3000.0, 600)
    p = canonical_params(ripplegw.waveform("IMRPhenomPv2_NRTidalv2"))
    tapered = ripplegw.waveform("IMRPhenomPv2_NRTidalv2")(f, p)["p"]
    free = ripplegw.waveform("IMRPhenomPv2_NRTidalv2", no_taper=True)(f, p)["p"]
    np.testing.assert_allclose(tapered[:50], free[:50], rtol=1e-12)
    assert np.abs(free[-1]) > 0 and np.abs(tapered[-1]) == 0.0


def test_lambda_tilde_parametrisation_matches_lambdas():
    f = jnp.linspace(25.0, 900.0, 100)
    plain = ripplegw.waveform("IMRPhenomPv2_NRTidalv2")
    tilde = ripplegw.waveform("IMRPhenomPv2_NRTidalv2", use_lambda_tildes=True)
    p = canonical_params(plain)
    pt = canonical_params(tilde)
    np.testing.assert_allclose(tilde(f, pt)["p"], plain(f, p)["p"], rtol=1e-8)
